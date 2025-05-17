class AIPlayer extends Player {
    constructor(position, velocity, mass, color) {
        // Call parent constructor
        super(position, velocity, mass, color);

        // --- AI Configuration ---
        // Distances & Ranges
        this.TARGET_SEARCH_RANGE = 800;         // Max distance to even look for any opponent
        this.FOOD_SEARCH_RANGE = 500;           // Max distance to look for food
        this.FIRING_RANGE = 450;                // Distance within which the AI will attempt to fire
        this.MIN_ENGAGE_DISTANCE = 50;          // Minimum distance to keep from target (avoids ramming)
        this.PLAYER_PRIORITY_RANGE = 10;      // Max distance to specifically prioritize human player
        this.ENGAGEMENT_RADIUS_MULTIPLIER = 10; // *** NEW: Engage targets within 10x own radius ***

        // Timers & Cooldowns
        this.thinkTimer = 0;              // Timer for decision making
        this.THINK_INTERVAL = 0.2;        // How often the AI re-evaluates (seconds)
        this.movementTimer = 0;           // Cooldown for movement actions
        this.MOVEMENT_COOLDOWN = 0.1;     // Short cooldown between move attempts
        this.shotDelayTimer = 0;          // Timer for delayed shots after aiming
        this.targetToShoot = null;        // Target locked for delayed shot

        // State Tracking
        this.currentTarget = null;        // The player (human or AI) currently targeted
        this.targetFood = null;           // The food item currently targeted
        this.currentState = 'IDLE';       // Current behavior state (IDLE, SEEK_FOOD, ATTACKING)
        this.avoidTarget = null;          // Temporary target point to move away from boundaries
        this.fleeTimer = 0;               // Timer for how long to flee
        this.lastAttacker = null;         // Who last hit the shield
        this.isFacingAway = false;        // NEW: Flag to indicate if bot is intentionally facing away
        this.faceAwayTimer = 0;           // NEW: Timer for how long to face away

        // Inherited from Player (relevant for AI)
        // this.position, this.velocity, this.mass, this.color, this.radius
        // this.fireCooldown, this.beamTimer, this.alive
        // this.sprite (set via setSprite)

        // Ensure AI has a base mass
        this.mass = Math.max(this.mass, 50); // Ensure minimum mass of 50
        this.updateRadius(); // Calculate initial radius based on mass

        // Match player speed constants (ensure these are defined in a Constants object/file)
        this.SPEED = Constants.BASE_SPEED || 200; // Use constant or default
        this.ACCELERATION_COST = Constants.ACCELERATION_COST || 0.05;
        this.accelerating = false; // AI doesn't use acceleration in this strategy

        // Rotation control properties (inherited/used from Player/CelestialBody)
        this.targetRotation = 0; // Target angle to face
        this.previousPosition = null; // Store position from last frame
    }

    // --- Core Update Loop ---
    update(dt, planets, foodList) {
        if (!this.alive) return; // Skip updates if dead

        // Basic physics and updates from parent
        super.update(dt, planets); // Handles movement, wrapping, cooldowns, radius updates if mass changes

        // --- Stuck Check & Random Rotation ---
        // Check if moved significantly less than expected speed
        this.checkIfStuckAndRotate(dt);
        // --- End Stuck Check ---

        // Update internal timers
        this.thinkTimer -= dt;
        if (this.movementTimer > 0) this.movementTimer -= dt;
        if (this.shotDelayTimer > 0) this.processShots(dt); // Handle delayed shots

        // NEW: Process faceAwayTimer
        if (this.faceAwayTimer > 0) {
            this.faceAwayTimer -= dt;
            if (this.faceAwayTimer <= 0) {
                this.isFacingAway = false;
                this.faceAwayTimer = 0;
            }
        }

        // --- AI Decision Making ---
        if (this.thinkTimer <= 0) {
            this.thinkTimer = this.THINK_INTERVAL; // Reset timer
            this.makeDecisions(planets, foodList); // Decide next action/state
        }

        // --- Execute Current State's Actions ---
        this.executeCurrentState(dt);

        // AI collects food passively if it touches it, regardless of state
        if (window.gameScene && window.gameScene.foodList) {
             this.collectFood(window.gameScene.foodList, window.gameScene.foodSprites, window.gameScene);
        }

        // --- Override Rotation: Always Face Player ---
        if (this.currentTarget === null && this.currentState !== 'AVOIDING_BOUNDARY') {
            this.currentTarget = window.gameScene ? window.gameScene.player : null; // Default to player if no target
        }

        if (this.currentTarget) {
            // Calculate direction vector to player, handling wrapping
            let dx = this.currentTarget.position.x - this.position.x;
            let dy = this.currentTarget.position.y - this.position.y;
            if (Math.abs(dx) > Constants.WIDTH / 2) dx = dx > 0 ? dx - Constants.WIDTH : dx + Constants.WIDTH;
            if (Math.abs(dy) > Constants.HEIGHT / 2) dy = dy > 0 ? dy - Constants.HEIGHT : dy + Constants.HEIGHT;

            let distanceTotatfedarget = Math.sqrt(dx * dx + dy * dy);

            if (distanceTotatfedarget > this.radius * 20){
                this.currentTarget = null; // Clear target if too far
                this.currentState = 'IDLE'; // Revert to IDLE if target lost
            } 

            // Set target rotation to face the player (assumes sprite faces RIGHT at rotation 0)
            this.targetRotation = Math.atan2(dy, dx) - Math.PI / 2; // Adjust for sprite orientation
            if (this.isFacingAway) {
                this.targetRotation += Math.PI; // Add 180 degrees if facing away
            }

            // MODIFIED: mult calculation based on state
            let mult;
            if (this.currentState === 'ATTACKING') {
                mult = 1; // When attacking, face away if target is within 1 radius
            } else {
                mult = Math.floor(Math.random() * 3) + 3; // Random integer between 3 and 5 for other states
            }

            if (distanceTotatfedarget < this.radius * mult){
                // Start facing away if not already doing so
                if (!this.isFacingAway) {
                    this.isFacingAway = true;
                    // Set timer for 0.5 to 1.5 seconds
                    this.faceAwayTimer = 0.5 + Math.random();

                    // Calculate the angle directly TO the target
                    const angleToTarget = Math.atan2(dy, dx);

                    // Set the target rotation to be 180 degrees opposite the target angle
                    // Adjust by -PI/2 because the sprite faces right at rotation 0 (consistent with above)
                    // This effectively sets the "away" rotation directly
                    this.targetRotation = (angleToTarget - Math.PI / 2) + Math.PI;

                    // Apply the rotation immediately so it doesn't wait for the next updateRotation call
                    this.updateRotation(0);

                }
            }
        }

        // Update rotation
        this.updateRotation(dt);

        // Store current position for next frame's stuck check
        this.previousPosition = { x: this.position.x, y: this.position.y };
    }

    // --- Decision Logic ---
    makeDecisions(planets, foodList) {
        const humanPlayer = window.gameScene ? window.gameScene.player : null;
        let potentialTarget = null;
        const engagementDistance = this.radius * this.ENGAGEMENT_RADIUS_MULTIPLIER; // Calculate engagement distance based on current radius
        const engagementDistanceSq = engagementDistance * engagementDistance; // Use squared distance

        if (this.currentTarget && !this.currentTarget.alive) {
            this.currentTarget = null; // Clear target if not alive
        }

        // --- Boundary Avoidance Check (Highest Priority) ---
        const boundaryAvoidDir = this.checkBoundaryProximity();
        if (boundaryAvoidDir) {
            // Only switch state if not already avoiding (prevents constant velocity reversal)
            if (this.currentState !== 'AVOIDING_BOUNDARY' && this.currentState !== 'FLEEING') {
                this.targetRotation += Math.PI; // Add 180 degrees (PI radians)
                this.updateRotation(0); // Apply rotation immediately

                this.currentState = 'AVOIDING_BOUNDARY';
                this.currentTarget = null; // Clear other targets
                this.targetFood = null;
                // No need for avoidTarget with this strategy
            }
            return; // Prioritize avoiding boundary
        } else {
            // If not near boundary, revert to IDLE state
            if (this.currentState === 'AVOIDING_BOUNDARY') {
                this.currentState = 'IDLE';
            }
        }

        // --- Prioritize Human Player (if within engagement range) ---
        if (humanPlayer && humanPlayer.alive) {
            const distSqToPlayer = this.distanceToTargetSq(humanPlayer);
            // Check if player is within BOTH the priority range AND the engagement range
            if (distSqToPlayer < this.PLAYER_PRIORITY_RANGE * this.PLAYER_PRIORITY_RANGE &&
                distSqToPlayer < engagementDistanceSq) {
                potentialTarget = humanPlayer; // Prioritize player if close enough
            }
        }

        // --- Find Closest Opponent (AI or Player if not already prioritized, must be within engagement range) ---
        if (!potentialTarget) {
            // Pass the calculated engagement distance to the finder function
            potentialTarget = this.findClosestOpponent(engagementDistance); // Only find opponents within engagement distance
        }

        // --- Set State based on Target ---
        if (potentialTarget) {
            // Found a valid target within engagement range
            this.currentState = 'ATTACKING';
            this.currentTarget = potentialTarget;
            this.targetFood = null; // Stop targeting food if attacking
        } else {
            // --- No Opponent Targeted within engagement range: Look for Food ---
            this.currentTarget = null; // Clear opponent target
            this.targetFood = this.findBestFood(foodList, this.FOOD_SEARCH_RANGE);

            if (this.targetFood) {
                this.currentState = 'SEEK_FOOD';
            } else {
                // --- No Opponent or Food: Idle/Wander ---
                this.currentState = 'IDLE';
            }
        }
    }

    // --- State Execution ---
    executeCurrentState(dt) {
        switch (this.currentState) {
            case 'ATTACKING':
                this.executeAttackStrategy(dt);
                break;
            case 'SEEK_FOOD':
                this.executeSeekFoodStrategy(dt);
                break;
            case 'AVOIDING_BOUNDARY':
                this.executeAvoidBoundary(dt);
                break;
            case 'FLEEING': // *** NEW STATE ***
                this.executeFleeingStrategy(dt);
                break;
            case 'IDLE':
            default:
                this.executeWander(dt); // Call wander method when idle
                break;
        }
    }

    // --- ATTACKING State Logic ---
    executeAttackStrategy(dt) {
        // Check if the target is still valid
        if (!this.currentTarget || !this.currentTarget.alive) {
            this.currentTarget = null;
            this.currentState = 'IDLE'; // Revert to IDLE if target lost
            return;
        }

        // Recalculate engagement distance in case radius changed
        const engagementDistance = this.radius * this.ENGAGEMENT_RADIUS_MULTIPLIER;
        const targetPos = this.currentTarget.position;
        const distance = this.distanceToTarget(this.currentTarget);

        // *** NEW: Check if target moved outside engagement range ***
        if (distance > engagementDistance) {
            this.currentTarget = null;
            this.currentState = 'IDLE'; // Target is too far, disengage
            return;
        }

        // 1. Aim: Always face the target when attacking
        this.facePoint(targetPos);

        // 2. Adjust distance: Move towards or maintain distance WITH planets parameter
        if (distance > this.FIRING_RANGE) {
            // Get planets from game scene
            const planets = window.gameScene ? window.gameScene.planetSprites : [];
            this.moveTowards(targetPos, dt, planets); // Pass planets parameter here
        } else if (distance < this.MIN_ENGAGE_DISTANCE) {
            this.velocity.set(0, 0);
        } else {
            this.velocity.set(0, 0);
        }

        // 3. Fire: Attempt to shoot if conditions are met
        if (distance <= this.FIRING_RANGE &&
            this.fireCooldown <= 0 &&
            this.shotDelayTimer <= 0 &&
            this.mass > Constants.FIRE_COST) {
            this.aimAndShoot(this.currentTarget); // Initiate delayed shot
        }
    }

    // --- SEEK_FOOD State Logic ---
    executeSeekFoodStrategy(dt) {
        // Check if the food target is still valid (exists in the main food list)
        if (!this.targetFood || !window.gameScene || !window.gameScene.foodList.includes(this.targetFood)) {
            this.targetFood = null;
            this.currentState = 'IDLE'; // Revert to IDLE if food lost
            return;
        }

        // Get planets from game scene
        const planets = window.gameScene ? window.gameScene.planetSprites : [];

        // Move towards the food target WITH planets parameter
        this.facePoint(this.targetFood.position);
        this.moveTowards(this.targetFood.position, dt, planets); // Pass planets parameter here

        // Note: Actual collection happens passively in the main update loop via `collectFood`
    }

    // --- AVOIDING_BOUNDARY State Logic ---
    executeAvoidBoundary(dt) {
        // First, check if we are still near a boundary
        const boundaryAvoidDir = this.checkBoundaryProximity();
        if (!boundaryAvoidDir) {
            // No longer near boundary, switch back to IDLE to re-evaluate
            this.currentState = 'IDLE';
            return;
        }

        const forwardX = Math.cos(this.targetRotation - Math.PI / 2);
        const forwardY = Math.sin(this.targetRotation - Math.PI / 2);
        this.velocity.x = forwardX * this.SPEED;
        this.velocity.y = forwardY * this.SPEED;
    }

    // --- FLEEING State Logic ---
    executeFleeingStrategy(dt) {
        const FLEE_DURATION = 2.0 + Math.random(); // Flee for 2-3 seconds

        this.fleeTimer -= dt;
        if (this.fleeTimer <= 0) {
            // Stop fleeing and re-evaluate
            this.fleeTimer = 0;
            this.accelerating = false;

            // If there is a valid last attacker, switch to ATTACKING
            if (this.lastAttacker && this.lastAttacker.alive) {
                this.currentState = 'ATTACKING';
                this.currentTarget = this.lastAttacker;
                this.lastAttacker = null; // Clear last attacker after switching to ATTACKING
                console.log(`AI ${this.name} is now attacking ${this.currentTarget.name || 'Unknown'} after fleeing.`);
            } else {
                // If no valid attacker, revert to IDLE
                this.currentState = 'IDLE';
                this.currentTarget = null;
            }
            return;
        }

        let fleeAngle;
        if (this.lastAttacker) {
            // Flee directly away from the last attacker
            fleeAngle = Math.atan2(
                this.lastAttacker.position.y - this.position.y,
                this.lastAttacker.position.x - this.position.x
            ) + Math.PI; // Add 180 degrees
        } else {
            // Flee in a random direction if attacker unknown
            fleeAngle = Math.random() * Math.PI * 2;
        }

        // Set rotation and move
        this.targetRotation = fleeAngle + Math.PI / 2; // Adjust based on sprite orientation
        this.updateRotation(0); // Apply rotation immediately
        this.velocity.x = Math.cos(fleeAngle) * this.SPEED;
        this.velocity.y = Math.sin(fleeAngle) * this.SPEED;
    }

    // --- Helper Methods ---

    /**
     * Finds the closest opponent (human or AI) **within the engagement range**.
     * @param {number} engagementRange - The maximum distance (based on radius * multiplier) to consider a target.
     * @returns {Player | AIPlayer | null} The closest valid opponent or null if none found.
     */
    findClosestOpponent(engagementRange) {
        if (!window.gameScene) return null;
        let closestTarget = null;
        // Use engagementRange squared for comparison
        let minDistanceSq = engagementRange * engagementRange;

        // List potential targets (human + other AIs)
        const potentialTargets = [
            window.gameScene.player,
            ...(window.gameScene.aiPlayers || [])
        ].filter(p => p && p.alive && p !== this); // Filter valid targets

        // Find the closest one *within* the engagement range
        for (const target of potentialTargets) {
            const distSq = this.distanceToTargetSq(target);
            // Check if within engagement range AND closer than current best
            if (distSq < minDistanceSq) {
                minDistanceSq = distSq;
                closestTarget = target;
            }
        }
        // Return the closest target found within the range, or null
        return closestTarget;
    }

    /**
     * Finds the best food item to target within a given range.
     * Prioritizes closer food items.
     * @param {Array<Food>} foodList - The list of available food items.
     * @param {number} range - The maximum distance to search for food.
     * @returns {Food | null} The best food target or null if none found.
     */
    findBestFood(foodList, range) {
        let bestFood = null;
        let minDistanceSq = range * range;

        if (!foodList) return null;

        for (const food of foodList) {
            if (!food || !food.position) continue; // Skip invalid food
            const distSq = this.distanceToTargetSq(food); // Use squared distance
            if (distSq < minDistanceSq) {
                minDistanceSq = distSq;
                bestFood = food;
            }
        }
        return bestFood;
    }


    /**
     * Calculates the squared distance to a target, handling world wrapping.
     * @param {object} target - The target object with a 'position' property {x, y}.
     * @returns {number} The squared distance to the target.
     */
    distanceToTargetSq(target) {
        if (!target || !target.position) return Infinity;
        let dx = target.position.x - this.position.x;
        let dy = target.position.y - this.position.y;

        // Account for world wrapping
        if (Math.abs(dx) > Constants.WIDTH / 2) {
            dx = dx > 0 ? dx - Constants.WIDTH : dx + Constants.WIDTH;
        }
        if (Math.abs(dy) > Constants.HEIGHT / 2) {
            dy = dy > 0 ? dy - Constants.HEIGHT : dy + Constants.HEIGHT;
        }
        return dx * dx + dy * dy; // Return squared distance
    }

    /**
     * Calculates the actual distance to a target, handling world wrapping.
     * @param {object} target - The target object with a 'position' property {x, y}.
     * @returns {number} The distance to the target.
     */
    distanceToTarget(target) {
        return Math.sqrt(this.distanceToTargetSq(target));
    }

    /**
     * Implements gradual rotation by creating intermediate arc points
     * @param {object} targetPosition - The final position to face
     */
    facePoint(targetPosition) {
        if (!this.sprite) return;

        // Calculate direction vector to target
        let dx = targetPosition.x - this.position.x;
        let dy = targetPosition.y - this.position.y;

        // Handle wrapping
        if (Math.abs(dx) > Constants.WIDTH / 2) {
            dx = dx > 0 ? dx - Constants.WIDTH : dx + Constants.WIDTH;
        }
        if (Math.abs(dy) > Constants.HEIGHT / 2) {
            dy = dy > 0 ? dy - Constants.HEIGHT : dy + Constants.HEIGHT;
        }

        // Calculate distance to target
        const distanceToTarget = Math.sqrt(dx * dx + dy * dy);

        // Get current facing direction vector (from sprite's rotation)
        const currentAngle = this.sprite.rotation - Math.PI/2; // Adjust for sprite orientation
        const currentDirX = Math.cos(currentAngle);
        const currentDirY = Math.sin(currentAngle);

        // Project a point at same distance in current facing direction
        const projectedPoint = {
            x: this.position.x + currentDirX * distanceToTarget,
            y: this.position.y + currentDirY * distanceToTarget
        };

        // Calculate intermediate point 1/10 of the way from projected to target
        const intermediatePoint = {
            x: projectedPoint.x + (targetPosition.x - projectedPoint.x) * 0.1,
            y: projectedPoint.y + (targetPosition.y - projectedPoint.y) * 0.1
        };

        // Calculate new direction to intermediate point
        let newDx = intermediatePoint.x - this.position.x;
        let newDy = intermediatePoint.y - this.position.y;

        // Set target rotation to face the intermediate point
        this.targetRotation = Math.atan2(newDy, newDx) + Math.PI/2;
        this.updateRotation(0);

        // Let updateRotation handle the actual sprite rotation
    }

    /**
     * Checks if a straight path intersects with any planet.
     * @param {Vector2} startPos - Starting position
     * @param {Vector2} endPos - Target position
     * @param {Array} planets - List of planets to check against
     * @returns {Object|null} The first planet that intersects with the path, or null if no intersection
     */
    isPathBlockedByPlanet(startPos, endPos, planets) {
        if (!planets || planets.length === 0) return null;

        // Calculate direction vector
        let dx = endPos.x - startPos.x;
        let dy = endPos.y - startPos.y;

        // Handle wrapping
        if (Math.abs(dx) > Constants.WIDTH / 2) {
            dx = dx > 0 ? dx - Constants.WIDTH : dx + Constants.WIDTH;
        }
        if (Math.abs(dy) > Constants.HEIGHT / 2) {
            dy = dy > 0 ? dy - Constants.HEIGHT : dy + Constants.HEIGHT;
        }

        // Normalize direction
        const distance = Math.sqrt(dx * dx + dy * dy);
        if (distance < 1) return null; // Too close to check

        const dirX = dx / distance;
        const dirY = dy / distance;

        // Check against each planet
        for (const planetData of planets) {
            const planet = planetData.planet;

            // Get vector from start to planet center
            let toPlanetX = planet.position.x - startPos.x;
            let toPlanetY = planet.position.y - startPos.y;

            // Handle wrapping
            if (Math.abs(toPlanetX) > Constants.WIDTH / 2) {
                toPlanetX = toPlanetX > 0 ? toPlanetX - Constants.WIDTH : toPlanetX + Constants.WIDTH;
            }
            if (Math.abs(toPlanetY) > Constants.HEIGHT / 2) {
                toPlanetY = toPlanetY > 0 ? toPlanetY - Constants.HEIGHT : toPlanetY + Constants.HEIGHT;
            }

            // Calculate projection of planet position onto path direction
            const projectionLength = toPlanetX * dirX + toPlanetY * dirY;

            // If projection is negative, planet is behind the start point
            // If projection is greater than distance, planet is past the end point
            if (projectionLength < 0 || projectionLength > distance) {
                continue;
            }

            // Calculate closest distance from path to planet center
            const projectedX = startPos.x + dirX * projectionLength;
            const projectedY = startPos.y + dirY * projectionLength;

            const closestDistSq =
                Math.pow(projectedX - (startPos.x + toPlanetX), 2) +
                Math.pow(projectedY - (startPos.y + toPlanetY), 2);

            // Check if path is too close to planet (considering planet radius + buffer)
            const safeDistance = planet.radius + this.radius + 10;
            if (closestDistSq < safeDistance * safeDistance) {
                return planet; // Path is blocked by this planet
            }
        }

        return null; // No blocking planets
    }

    /**
     * Calculates a detour point to avoid a planet in the path.
     * @param {Vector2} targetPos - The original target position
     * @param {Object} blockingPlanet - The planet blocking the path
     * @returns {Vector2} A new intermediate target position that avoids the planet
     */
    calculateDetourPoint(targetPos, blockingPlanet) {
        // Get vectors from bot to planet and bot to target
        let toPlanetX = blockingPlanet.position.x - this.position.x;
        let toPlanetY = blockingPlanet.position.y - this.position.y;
        let toTargetX = targetPos.x - this.position.x;
        let toTargetY = targetPos.y - this.position.y;

        // Handle wrapping
        if (Math.abs(toPlanetX) > Constants.WIDTH / 2) {
            toPlanetX = toPlanetX > 0 ? toPlanetX - Constants.WIDTH : toPlanetX + Constants.WIDTH;
        }
        if (Math.abs(toPlanetY) > Constants.HEIGHT / 2) {
            toPlanetY = toPlanetY > 0 ? toPlanetY - Constants.HEIGHT : toPlanetY + Constants.HEIGHT;
        }

        // Normalize planet vector
        const planetDist = Math.sqrt(toPlanetX * toPlanetX + toPlanetY * toPlanetY);
        const normPlanetX = toPlanetX / planetDist;
        const normPlanetY = toPlanetY / planetDist;

        // Calculate perpendicular vector to the planet direction
        const perpX = -normPlanetY;
        const perpY = normPlanetX;

        // Determine which side of the planet to pass by (dot product with target vector)
        const dotProduct = perpX * toTargetX + perpY * toTargetY;
        const sideMultiplier = dotProduct > 0 ? 1 : -1;

        // Calculate safe distance to pass by
        const safeDistance = blockingPlanet.radius + this.radius * 2 + 30;

        // Create detour point that goes around the planet
        return {
            x: blockingPlanet.position.x + perpX * safeDistance * sideMultiplier,
            y: blockingPlanet.position.y + perpY * safeDistance * sideMultiplier
        };
    }

    /**
     * Moves the AI towards a target position by setting its velocity.
     * @param {object} targetPosition - The position to move towards {x, y}.
     * @param {number} dt - Delta time.
     * @param {Array} planets - List of planets to check for path blocking.
     */
    moveTowards(targetPosition, dt, planets = []) {
        if (this.movementTimer > 0 || !this.sprite) return;

        // Check if path to target is blocked by a planet
        const blockingPlanet = this.isPathBlockedByPlanet(this.position, targetPosition, planets);

        // If blocked, calculate detour
        if (blockingPlanet) {
            // Use detour point instead of original target
            targetPosition = this.calculateDetourPoint(targetPosition, blockingPlanet);
        }

        // Continue with existing movement logic
        let dx = targetPosition.x - this.position.x;
        let dy = targetPosition.y - this.position.y;

        if (Math.abs(dx) > Constants.WIDTH / 2) {
            dx = dx > 0 ? dx - Constants.WIDTH : dx + Constants.WIDTH;
        }
        if (Math.abs(dy) > Constants.HEIGHT / 2) {
            dy = dy > 0 ? dy - Constants.HEIGHT : dy + Constants.HEIGHT;
        }

        const distance = Math.sqrt(dx * dx + dy * dy);
        if (distance < 1) {
            this.velocity.set(0, 0);
            return;
        }

        // Normalize direction
        const directionX = dx / distance;
        const directionY = dy / distance;

        // Set velocity towards target
        this.velocity.x = directionX * this.SPEED;
        this.velocity.y = directionY * this.SPEED;

        // Reset movement cooldown
        this.movementTimer = this.MOVEMENT_COOLDOWN;
    }

    /**
     * Aims at the target and sets up a delayed shot timer.
     * @param {Player | AIPlayer} target - The target to shoot at.
     */
    aimAndShoot(target) {
        // Check conditions before initiating shot sequence
        if (!target || !target.alive || this.shotDelayTimer > 0 || this.fireCooldown > 0 || this.mass <= Constants.FIRE_COST) {
            return;
        }
        // Aim at the target
        this.facePoint(target.position);
        // Set a random delay for the shot
        this.shotDelayTimer = 0.1 + Math.random() * 0.2;
        // Store the target
        this.targetToShoot = target;
    }

    /**
     * Processes the delayed shot timer and fires if ready.
     * @param {number} dt - Delta time.
     */
    processShots(dt) {
        if (this.shotDelayTimer > 0) {
            this.shotDelayTimer -= dt;
            if (this.shotDelayTimer <= 0) {
                this.shotDelayTimer = 0; // Reset timer

                // Final check before firing
                if (this.targetToShoot && this.targetToShoot.alive &&
                    this.mass > Constants.FIRE_COST && this.fireCooldown <= 0) {

                    // First, ensure target rotation is set correctly
                    this.facePoint(this.targetToShoot.position);

                    // Apply the rotation immediately before firing
                    this.updateRotation(0);

                    // Now fire in the correctly facing direction
                    this.fireBeam(window.gameScene);
                }

                // Clear target regardless
                this.targetToShoot = null;
            }
        }
    }

    /**
     * Updates the rotation of the AI sprite to instantly face the target rotation.
     * Since facePoint() already calculates intermediate points for gradual rotation,
     * we don't need additional smoothing here.
     * @param {number} dt - Delta time.
     */
    updateRotation(dt) {
        if (!this.sprite || this.targetRotation === undefined) return;

        // Instantly set rotation to target rotation
        this.sprite.rotation = this.targetRotation;
    }

    /**
     * Checks if the bot is close to a boundary and returns a vector pointing away from it.
     * @returns {object|null} A normalized vector {x, y} pointing away from the boundary, or null if not near.
     */
    checkBoundaryProximity() {
        const threshold = this.radius + 30; // Distance from edge to trigger avoidance
        let avoidX = 0;
        let avoidY = 0;

        if (this.position.x < threshold) avoidX = 1;
        else if (this.position.x > Constants.WIDTH - threshold) avoidX = -1;

        if (this.position.y < threshold) avoidY = 1;
        else if (this.position.y > Constants.HEIGHT - threshold) avoidY = -1;

        if (avoidX !== 0 || avoidY !== 0) {
            // Normalize the avoidance vector
            const len = Math.sqrt(avoidX * avoidX + avoidY * avoidY);
            return { x: avoidX / len, y: avoidY / len };
        }

        return null; // Not near any boundary
    }

    /**
     * Checks if the bot moved significantly less than its speed suggests
     * and applies a random rotation if it seems stuck.
     * @param {number} dt Delta time for the frame.
     */
    checkIfStuckAndRotate(dt) {
        if (!this.previousPosition || dt <= 0) {
            return; // Cannot check on first frame or if dt is zero
        }

        // Calculate squared distance moved since last frame
        const dx = this.position.x - this.previousPosition.x;
        const dy = this.position.y - this.previousPosition.y;
        const actualDistSq = dx * dx + dy * dy;

        // Calculate expected distance based on speed
        const expectedDist = this.SPEED * dt;
        // Set a threshold (e.g., moved less than 30% of expected distance)
        const thresholdDistSq = (expectedDist * 0.9) * (expectedDist * 0.9);

        if (actualDistSq < thresholdDistSq) {
            // Apply a random rotation (e.g., up to +/- 60 degrees)
            this.targetRotation += (Math.random() - 0.5) * (Math.PI / 1.5);
            this.updateRotation(0); // Apply rotation immediately
        }
    }

    // --- Overrides or Placeholder Methods ---

    /**
     * Overrides the parent die method to add AI-specific cleanup.
     */
    die(reason = null, killer = null) {
        if (!this.alive) return;
        super.die(reason, killer); // Call parent cleanup
        // Reset AI state
        this.currentState = 'IDLE';
        this.currentTarget = null;
        this.targetFood = null;
        this.targetToShoot = null;
        this.shotDelayTimer = 0;
    }

    /**
     * Implements wandering behavior when the AI is in the IDLE state.
     * @param {number} dt - Delta time.
     */
    executeWander(dt) {
        // Only initiate new wander movement if cooldown is over
        if (this.movementTimer <= 0) {
            const angle = Math.random() * Math.PI * 2; // Pick random direction
            // Define a point slightly ahead in that direction
            const randomTargetPos = {
                x: this.position.x + Math.cos(angle) * 100,
                y: this.position.y + Math.sin(angle) * 100
            };
            this.facePoint(randomTargetPos); // Face the random point

            // Get planets from game scene
            const planets = window.gameScene ? window.gameScene.planetSprites : [];

            // Move WITH planets parameter
            this.moveTowards(randomTargetPos, dt, planets); // Pass planets parameter here
        }
        // If IDLE and not actively moving towards a wander point, stop moving
        else if(this.currentState === 'IDLE' && this.movementTimer <= 0) {
             this.velocity.set(0,0);
         }
    }

    /**
     * Called by the Shield when it takes damage.
     * @param {Player | AIPlayer} attacker - Who fired the beam.
     */
    handleShieldHit(attacker) {
        // *** NEW: Ignore shield hit if already attacking ***
        if (this.currentState === 'ATTACKING') {
            console.log(`AI ${this.name} is already ATTACKING, ignoring shield hit from ${attacker ? attacker.name : 'Unknown'}`);
            return;
        }

        // Don't interrupt boundary avoidance
        if (this.currentState === 'AVOIDING_BOUNDARY') {
            return;
        }

        // Prioritize Attacking Back
        if (attacker && attacker.alive) {
            this.currentState = 'ATTACKING';
            this.currentTarget = attacker;
            this.targetFood = null; // Stop targeting food
            this.fleeTimer = 0; // Stop any fleeing
            this.lastAttacker = null; // Clear last attacker if we are now attacking them
            // Reset think timer to give attack strategy a chance to run immediately
            this.thinkTimer = this.THINK_INTERVAL;
            console.log(`AI ${this.name} is retaliating against ${attacker.name || 'Attacker'}`);
        } else {
            // If attacker is unknown or dead, then flee as a fallback
            const FLEE_DURATION = 2.0; // Flee for 2 seconds
            this.currentState = 'FLEEING';
            this.fleeTimer = FLEE_DURATION;
            this.lastAttacker = attacker; // Store for flee logic even if null
            this.accelerating = true;
            this.currentTarget = null;
            this.targetFood = null;
        }
    }

    // Assumes necessary methods like fireBeam, facePoint, collectFood, updateRadius etc.
    // are inherited from Player or defined in the Player class.
}

// Ensure the game scene reference is globally accessible
// window.gameScene = this; // Should be set in the GameScene's create method