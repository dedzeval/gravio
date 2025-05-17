import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import argparse
from matplotlib.path import Path

def generate_pentagram_points(radius=1, rotation=0):
    """Generate points for a 5-point star (pentagram)"""
    angles = np.linspace(0, 2 * np.pi, 5, endpoint=False) + rotation
    points = np.array([np.cos(angles), np.sin(angles)]).T * radius
    # Order to form a pentagram: connect every second point
    order = [0, 2, 4, 1, 3]
    return points[order]

def generate_rotated_star_image(frame_index, num_frames, image_size=64):
    """Generate a single star frame at the specified rotation index"""
    # Calculate rotation angle for this frame (0 to 2π)
    angle = frame_index * (2 * np.pi / num_frames)

    # Set up figure with transparent background
    fig = plt.figure(figsize=(1, 1), dpi=image_size, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])  # No padding
    ax.set_axis_off()
    fig.add_axes(ax)

    # Create initial star points (start with point straight up)
    initial_points_2d = generate_pentagram_points(radius=0.85, rotation=np.pi/2)

    # Convert to 3D points (z=0 for all points initially)
    initial_points_3d = np.zeros((len(initial_points_2d), 3))
    initial_points_3d[:, 0] = initial_points_2d[:, 0]  # x
    initial_points_3d[:, 1] = initial_points_2d[:, 1]  # y
    initial_points_3d[:, 2] = 0  # z=0 initially

    # Apply 3D rotation around y-axis
    rotated_points_3d = np.zeros_like(initial_points_3d)
    for i, point in enumerate(initial_points_3d):
        x, y, z = point

        # Apply rotation matrix around y-axis in 3D space
        rotated_points_3d[i, 0] = x * np.cos(angle) + z * np.sin(angle)  # x'
        rotated_points_3d[i, 1] = y                                      # y (unchanged)
        rotated_points_3d[i, 2] = -x * np.sin(angle) + z * np.cos(angle) # z'

    # Project back to 2D (perspective projection)
    # Using a simple perspective projection with the camera at (0,0,-3)
    projected_points = np.zeros((len(rotated_points_3d), 2))
    camera_z = -3
    for i, point in enumerate(rotated_points_3d):
        x, y, z = point
        # Perspective projection
        factor = camera_z / (camera_z - z)
        projected_points[i, 0] = x * factor
        projected_points[i, 1] = y * factor

    # Draw the outer border star (yellow)
    outer_star = patches.Polygon(projected_points, closed=True,
                                 edgecolor='#FFD700',  # Gold
                                 fill=False,
                                 linewidth=3)
    ax.add_patch(outer_star)

    # Create inner star points (slightly smaller)
    inner_scale = 0.8
    inner_points = projected_points * inner_scale

    # Draw the inner star (greenish)
    inner_star = patches.Polygon(inner_points, closed=True,
                                 edgecolor='#50C878',  # Emerald green
                                 fill=False,
                                 linewidth=1.5)
    ax.add_patch(inner_star)

    # Set limits to ensure the star fits tightly in view
    margin = 0.2
    ax.set_xlim(-1-margin, 1+margin)
    ax.set_ylim(-1-margin, 1+margin)

    return fig

def generate_star_frames(num_frames=12, image_size=64):
    """Generate a series of star frames and save them as PNGs"""
    # Create assets folder if it doesn't exist
    os.makedirs("assets", exist_ok=True)

    # Generate each frame
    for i in range(num_frames):
        fig = generate_rotated_star_image(i, num_frames, image_size)

        # Save as PNG with transparency
        filename = f"assets/star_{i:02d}.png"
        fig.savefig(filename, transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f"Saved {filename}")

    print(f"Generated {num_frames} star frames in assets/ directory")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate star rotation PNG frames")
    parser.add_argument("--size", type=int, default=64, help="Image size in pixels (default: 64)")
    parser.add_argument("--frames", type=int, default=12, help="Number of frames to generate (default: 12)")
    args = parser.parse_args()

    generate_star_frames(num_frames=args.frames, image_size=args.size)