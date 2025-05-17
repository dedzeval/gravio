# -*- coding: utf-8 -*-
"""
CNN-based Deep Q-Network (DQN) implementation using PyTorch
to train an agent, demonstrating handling of image-based input.

Uses Gymnasium environment wrappers on CartPole-v1 to simulate pixel input
for demonstration purposes while respecting macOS M3 (MPS) and < 4GB RAM constraints.

Requires: gymnasium, torch, numpy, opencv-python, matplotlib
Install opencv: pip install opencv-python
"""

import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.wrappers import FrameStack, GrayScaleObservation, ResizeObservation
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import time
import cv2 # For image processing in wrappers

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# --- Check for MPS availability (for M3 chip acceleration) ---
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    # Ensure MPS is available and built
    # Note: Performance might vary, CPU could be faster for very small nets
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    # Fallback to CUDA if available
    device = torch.device("cuda")
    print("Using CUDA")
else:
    # Fallback to CPU
    device = torch.device("cpu")
    print("Using CPU")

# --- Environment Setup & Preprocessing Wrappers ---
# We'll wrap CartPole to make it output "images"
# This is *not* standard for CartPole, just for demonstrating CNNs.

def create_wrapped_env(env_id="CartPole-v1", render_mode=None, img_size=84, num_stack=4):
    """Creates and wraps the environment for CNN input."""
    # Use render_mode="rgb_array" to get pixel data
    env = gym.make(env_id, render_mode="rgb_array")

    # 1. Resize Observation: Make frames smaller (e.g., 84x84)
    env = ResizeObservation(env, shape=img_size) # shape becomes (img_size, img_size)

    # 2. Grayscale Observation: Convert RGB to grayscale (reduces channels from 3 to 1)
    env = GrayScaleObservation(env, keep_dim=True) # keep_dim=True -> shape (img_size, img_size, 1)

    # 3. Frame Stacking: Stack consecutive frames to capture motion dynamics
    #    keep_dim=False stacks along a new first dimension: (num_stack, img_size, img_size, 1)
    #    We'll adjust the shape later for PyTorch's NCHW format.
    env = FrameStack(env, num_stack=num_stack, lz4_compress=True) # lz4_compress saves memory

    # Set render_mode for visualization if needed separately
    if render_mode == "human":
        env.render_mode = "human"

    print(f"Wrapped Env Observation Space: {env.observation_space}")
    print(f"Wrapped Env Action Space: {env.action_space}")
    return env

# --- Experience Replay Buffer ---
# Stores transitions (state, action, next_state, reward, done)
# State and next_state will now be stacked image frames.

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Saves a transition. Converts state/next_state numpy arrays to tensors."""
        state, action, next_state, reward, done = args

        # Convert numpy arrays (from FrameStack) to tensors here to save memory
        # Ensure correct dtype and move to device later during sampling if needed
        state_t = torch.tensor(np.array(state), dtype=torch.uint8) # Store as uint8 to save memory
        action_t = torch.tensor([action], dtype=torch.long)
        reward_t = torch.tensor([reward], dtype=torch.float32)
        done_t = torch.tensor([done], dtype=torch.bool)

        if next_state is not None:
            next_state_t = torch.tensor(np.array(next_state), dtype=torch.uint8) # Store as uint8
        else:
            next_state_t = None # Keep None marker for terminal states

        self.memory.append(Transition(state_t, action_t, next_state_t, reward_t, done_t))

    def sample(self, batch_size):
        """Samples a batch of transitions randomly."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# --- CNN Network Architecture ---
# Based on Nature DQN paper, adjusted slightly for simplicity/memory.
# Input shape expected by PyTorch Conv2d: (N, C, H, W)
# N: Batch size, C: Channels (num_stack), H: Height, W: Width

class CnnDQN(nn.Module):
    """Convolutional Deep Q-Network model."""
    def __init__(self, h, w, n_actions, num_stack=4):
        """
        Initializes the CnnDQN layers.
        Args:
            h (int): Height of the input image frames.
            w (int): Width of the input image frames.
            n_actions (int): Number of possible discrete actions.
            num_stack (int): Number of frames stacked (input channels).
        """
        super(CnnDQN, self).__init__()
        self.num_stack = num_stack
        self.h = h
        self.w = w

        # Convolutional layers
        self.conv1 = nn.Conv2d(num_stack, 32, kernel_size=8, stride=4) # Input channels = num_stack
        self.bn1 = nn.BatchNorm2d(32) # Batch norm for stability
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Function to calculate output size of conv layers
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64 # Flattened size from last conv layer

        print(f"CNN calculated linear input size: {linear_input_size}")
        if linear_input_size <= 0:
             raise ValueError("Calculated linear input size is not positive. Check conv layers/input size.")


        # Fully connected layers
        # Reduced size (e.g., 256) compared to original 512 for memory
        self.fc1 = nn.Linear(linear_input_size, 256)
        self.fc_out = nn.Linear(256, n_actions) # Output layer (Q-values per action)

    def forward(self, x):
        """
        Defines the forward pass of the network.
        Args:
            x (torch.Tensor): Input state tensor (N, C, H, W).
                               Expected dtype: float32, range [0, 1]
        Returns:
            torch.Tensor: Output tensor containing Q-values for each action.
        """
        # Apply convolutional layers with ReLU activation and batch norm
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)

        # Apply fully connected layers with ReLU activation
        x = F.relu(self.fc1(x))
        return self.fc_out(x) # Output raw Q-values

# --- Agent Class ---
# Encapsulates the CnnDQN networks, optimizer, memory, and learning logic.

class DQNAgent:
    """Agent that interacts with and learns from the environment using CNN."""

    def __init__(self, env, memory_capacity=10000,
                 batch_size=32, gamma=0.99, tau=0.005, lr=1e-4,
                 eps_start=1.0, eps_end=0.1, eps_decay=50000, img_size=84, num_stack=4):
        """
        Initializes the DQNAgent.
        Args:
            env (gym.Env): The (wrapped) environment instance.
            memory_capacity (int): Max size of the replay buffer.
            batch_size (int): Size of batches sampled from replay buffer. Reduced for CNNs.
            gamma (float): Discount factor.
            tau (float): Update rate for the target network (soft update).
            lr (float): Learning rate. Often lower for CNNs (e.g., 1e-4).
            eps_start (float): Starting epsilon. Often higher for complex tasks.
            eps_end (float): Minimum epsilon. Often higher for complex tasks.
            eps_decay (int): Slower epsilon decay for complex tasks.
            img_size (int): Height/Width of the observation image.
            num_stack (int): Number of frames stacked.
        """
        self.env = env
        self.n_actions = env.action_space.n
        self.img_size = img_size
        self.num_stack = num_stack
        # Observation space shape from FrameStack is (num_stack, H, W, 1) for GrayScale
        # We need (num_stack, H, W) for the network input C, H, W
        # self.obs_shape = env.observation_space.shape # (num_stack, H, W, 1)
        # print(f"Raw observation space shape: {self.obs_shape}")


        self.memory_capacity = memory_capacity
        self.batch_size = batch_size # Smaller batch size for CNNs due to memory
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay # Slower decay often needed for CNNs

        # Initialize Policy Network and Target Network
        self.policy_net = CnnDQN(img_size, img_size, self.n_actions, num_stack).to(device)
        self.target_net = CnnDQN(img_size, img_size, self.n_actions, num_stack).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.memory = ReplayMemory(self.memory_capacity)
        self.steps_done = 0
        self.episode_durations = []


    def _preprocess_state(self, state_batch):
        """
        Prepares a batch of states for the network.
        Converts uint8 [0, 255] to float32 [0.0, 1.0] and moves to device.
        Adjusts shape for PyTorch (N, C, H, W).
        Args:
            state_batch (torch.Tensor): Batch of states, shape (N, C, H, W, 1) or (N, C, H, W) from replay buffer.
                                        Stored as uint8.
        Returns:
            torch.Tensor: Processed batch, shape (N, C, H, W), float32, on device.
        """
        # Ensure state_batch is a tensor
        if not isinstance(state_batch, torch.Tensor):
             # This case handles single state inference
             state_batch = torch.tensor(np.array(state_batch), dtype=torch.uint8).unsqueeze(0) # Add batch dim

        # Squeeze the last dimension if it's 1 (from GrayScaleObservation keep_dim=True)
        if state_batch.dim() == 5 and state_batch.shape[-1] == 1:
            state_batch = state_batch.squeeze(-1) # Shape becomes (N, C, H, W)

        # Convert to float, normalize, and move to device
        processed_batch = state_batch.to(device, dtype=torch.float32) / 255.0
        return processed_batch


    def select_action(self, state):
        """
        Selects an action using an epsilon-greedy policy.
        Args:
            state (np.ndarray | LazyFrames): The current state observation from env.
        Returns:
            int: The selected action index.
        """
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1

        if sample > eps_threshold:
            # Exploitation
            with torch.no_grad():
                # Preprocess state: add batch dim, convert, normalize, move to device
                state_tensor = self._preprocess_state(state) # Shape (1, C, H, W)
                # Get Q-values from policy network
                q_values = self.policy_net(state_tensor)
                # Select action with highest Q-value
                action = q_values.max(1)[1].item() # Use .item() to get Python int
                return action
        else:
            # Exploration
            return self.env.action_space.sample()


    def optimize_model(self):
        """Performs one step of optimization on the policy network."""
        if len(self.memory) < self.batch_size:
            return # Not enough samples yet

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions)) # Convert batch of Transitions to Transition of batches

        # --- Prepare Batch Data ---
        # Filter out None values in next_state_batch (terminal states)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                       device=device, dtype=torch.bool)
        non_final_next_states_list = [s for s in batch.next_state if s is not None]

        # Stack tensors only if there are non-final states
        if len(non_final_next_states_list) > 0:
            non_final_next_states = torch.stack(non_final_next_states_list)
            # Preprocess next states
            non_final_next_states = self._preprocess_state(non_final_next_states) # (N_non_final, C, H, W)
        else:
             # Handle edge case where all sampled transitions are terminal
             # Create an empty tensor with the correct shape structure but 0 batch size
             non_final_next_states = torch.empty((0, self.num_stack, self.img_size, self.img_size),
                                                 device=device, dtype=torch.float32)


        # Stack and preprocess state, action, reward batches
        state_batch = torch.stack(batch.state)
        state_batch = self._preprocess_state(state_batch) # (B, C, H, W)

        action_batch = torch.cat(batch.action).unsqueeze(1).to(device) # Shape (B, 1)
        reward_batch = torch.cat(batch.reward).to(device)              # Shape (B,)

        # --- Compute Q(s_t, a) ---
        # Policy net predicts Q-values for all actions for the current states
        # Then we select the Q-value for the action actually taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch) # Shape (B, 1)

        # --- Compute V(s_{t+1}) for non-final next states ---
        # Target net predicts Q-values for all actions for the next states
        # We take the max Q-value (max_a' Q_target(s', a'))
        next_state_values = torch.zeros(self.batch_size, device=device) # Initialize with zeros
        if len(non_final_next_states_list) > 0:
             with torch.no_grad():
                 next_state_q_values = self.target_net(non_final_next_states)
                 next_state_values[non_final_mask] = next_state_q_values.max(1)[0] # Shape (N_non_final,)

        # --- Compute Expected Q Values (Target) ---
        # Expected Q = r + γ * V(s_{t+1})   (V is 0 for terminal states)
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch # Shape (B,)

        # --- Compute Loss ---
        criterion = nn.SmoothL1Loss()
        # Unsqueeze target to match shape of prediction (B, 1)
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # --- Optimize ---
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100) # Gradient clipping
        self.optimizer.step()


    def update_target_network(self):
        """Soft update of the target network's weights."""
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

    def plot_durations(self, show_result=False, label='CNN DQN'):
        """Plots the durations of episodes over time."""
        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        if show_result:
            plt.title(f'Result - {label}')
        else:
            plt.clf()
            plt.title(f'Training... - {label}')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy(), label='Episode Duration')
        # Plot rolling average
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy(), label='Avg Duration (100 episodes)')
        plt.legend()
        plt.pause(0.001)

    def save_model(self, path="cnndqn_cartpole_model.pth"):
        """Saves the policy network weights."""
        print(f"Saving model to {path}...")
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path="cnndqn_cartpole_model.pth"):
        """Loads the policy network weights."""
        print(f"Loading model from {path}...")
        try:
            # Load state dict, ensuring it's mapped to the correct device
            self.policy_net.load_state_dict(torch.load(path, map_location=device))
            self.policy_net.eval()
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
            print("Model loaded successfully.")
        except FileNotFoundError:
             print(f"Error: Model file not found at {path}. Training from scratch.")
        except Exception as e:
             print(f"Error loading model: {e}. Training from scratch.")


# --- Training Loop ---
def train(agent, env, num_episodes=500, optimize_freq=4, target_update_freq=1000):
    """
    Trains the CNN DQN agent.
    Args:
        agent (DQNAgent): The agent instance.
        env (gym.Env): The wrapped environment.
        num_episodes (int): Number of episodes to train for.
        optimize_freq (int): How often (in steps) to call optimize_model.
        target_update_freq (int): How often (in steps) to perform soft target update.
    """
    print("Starting Training (CNN DQN)...")
    start_time = time.time()
    total_steps = 0

    for i_episode in range(num_episodes):
        state, info = env.reset() # state is LazyFrames or ndarray
        episode_reward = 0
        terminated = False
        truncated = False
        duration = 0

        while not terminated and not truncated:
            duration += 1
            total_steps += 1

            action = agent.select_action(state) # state passed as is
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated

            # Store transition - use observation directly (LazyFrames or ndarray)
            # Conversion to tensor happens inside memory.push()
            next_state = observation if not done else None
            agent.memory.push(state, action, next_state, reward, done)

            # Move to the next state
            state = observation

            # Perform optimization periodically
            if total_steps > agent.batch_size and total_steps % optimize_freq == 0:
                 agent.optimize_model()

            # Soft update target network periodically
            if total_steps % target_update_freq == 0:
                 agent.update_target_network()

            if done:
                agent.episode_durations.append(duration)
                # agent.plot_durations(label='CNN DQN') # Optional: plot during training
                break

        if (i_episode + 1) % 20 == 0: # Print progress less frequently for longer training
            avg_duration = np.mean(agent.episode_durations[-20:]) if agent.episode_durations else 0
            print(f"Episode {i_episode+1}/{num_episodes} | Steps: {total_steps} | Avg Duration (Last 20): {avg_duration:.2f} | Epsilon: {agent.eps_end + (agent.eps_start - agent.eps_end) * math.exp(-1. * agent.steps_done / agent.eps_decay):.3f}")
            # Save model periodically
            agent.save_model()


    print('Training Complete')
    end_time = time.time()
    print(f"Training took: {end_time - start_time:.2f} seconds")
    agent.plot_durations(show_result=True, label='CNN DQN')
    plt.ioff()
    plt.show()


# --- Inference Function ---
def run_inference(agent, env_id="CartPole-v1", num_episodes=10, img_size=84, num_stack=4):
    """
    Runs the trained CNN agent in the environment without learning.
    Args:
        agent (DQNAgent): The trained agent (model should be loaded).
        env_id (str): ID of the base environment.
        num_episodes (int): Number of episodes to run.
        img_size (int): Image size used during training.
        num_stack (int): Frame stack size used during training.
    """
    print("\nStarting Inference (CNN DQN)...")
    agent.policy_net.eval() # Set model to evaluation mode

    # Create a new wrapped env for inference, potentially with human rendering
    inference_env = create_wrapped_env(env_id, render_mode="human", img_size=img_size, num_stack=num_stack)

    total_rewards = []
    for i_episode in range(num_episodes):
        state, info = inference_env.reset()
        terminated = False
        truncated = False
        episode_reward = 0
        duration = 0

        while not terminated and not truncated:
            duration += 1
            # Render the environment
            # Note: Rendering might be slow with wrappers
            try:
                 frame = inference_env.render()
            except Exception as e:
                 print(f"Rendering failed: {e}") # Handle potential rendering issues

            # Select action purely based on policy (no exploration)
            with torch.no_grad():
                 # Preprocess state (add batch dim, convert, normalize, move to device)
                 state_tensor = agent._preprocess_state(state) # Shape (1, C, H, W)
                 q_values = agent.policy_net(state_tensor)
                 action = q_values.max(1)[1].item()

            observation, reward, terminated, truncated, _ = inference_env.step(action)
            episode_reward += reward

            state = observation # Update state for next step

            if terminated or truncated:
                total_rewards.append(episode_reward)
                print(f"Inference Episode {i_episode+1}: Duration={duration}, Reward={episode_reward}")
                time.sleep(0.5) # Pause briefly

    inference_env.close()
    avg_reward = np.mean(total_rewards) if total_rewards else 0
    print(f"\nInference finished. Average reward over {num_episodes} episodes: {avg_reward:.2f}")


# --- Main Execution ---
if __name__ == "__main__":
    # --- Hyperparameters ---
    ENV_ID = "CartPole-v1"   # Base environment ID
    IMG_SIZE = 84            # Resize observations to 84x84
    NUM_STACK = 4            # Stack 4 consecutive frames

    BATCH_SIZE = 32          # Smaller batch size for CNN memory
    GAMMA = 0.99
    # Epsilon decay needs to be much slower for image-based tasks
    EPS_START = 1.0
    EPS_END = 0.1            # End exploration higher than for simple tasks
    EPS_DECAY = 30000        # Much slower decay (more steps to explore)
    TAU = 0.005
    LR = 1e-4                # Lower learning rate often better for CNNs
    # Memory capacity might need adjustment based on actual RAM usage
    MEMORY_CAPACITY = 50000  # Increased buffer size often helps CNNs
    NUM_EPISODES = 200       # Training duration (increase for real tasks)
    OPTIMIZE_FREQ = 4        # Optimize policy network every 4 steps
    TARGET_UPDATE_FREQ = 1000 # Soft update target network every 1000 steps

    # Create the wrapped environment
    # Use render_mode=None during training for speed
    train_env = create_wrapped_env(ENV_ID, render_mode=None, img_size=IMG_SIZE, num_stack=NUM_STACK)

    # Create the agent
    agent = DQNAgent(train_env,
                     memory_capacity=MEMORY_CAPACITY,
                     batch_size=BATCH_SIZE,
                     gamma=GAMMA,
                     tau=TAU,
                     lr=LR,
                     eps_start=EPS_START,
                     eps_end=EPS_END,
                     eps_decay=EPS_DECAY,
                     img_size=IMG_SIZE,
                     num_stack=NUM_STACK)

    # --- Choose Mode: Train or Inference ---
    MODE = "train" # Set to "train" or "inference"
    MODEL_PATH = "cnndqn_cartpole_model.pth" # Model filename

    if MODE == "train":
        # Optional: Load existing model to continue training
        # agent.load_model(MODEL_PATH)
        train(agent, train_env, num_episodes=NUM_EPISODES,
              optimize_freq=OPTIMIZE_FREQ, target_update_freq=TARGET_UPDATE_FREQ)
        # Save the trained model
        agent.save_model(MODEL_PATH)
        # Optional: Run inference after training
        run_inference(agent, ENV_ID, num_episodes=5, img_size=IMG_SIZE, num_stack=NUM_STACK)

    elif MODE == "inference":
        # Load the trained model
        agent.load_model(MODEL_PATH)
        # Run inference
        run_inference(agent, ENV_ID, num_episodes=10, img_size=IMG_SIZE, num_stack=NUM_STACK)

    else:
        print("Error: Invalid MODE selected. Choose 'train' or 'inference'.")

    train_env.close() # Close the training environment
    print("Done.")