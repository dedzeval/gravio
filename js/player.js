class Player extends CelestialBody {
    static MIN_MASS = 1;
    static PLAYER_INITIAL_MASS = 50;
    static PULL_ZONE_FACTOR = 2.5; // Pull zone is 2.5x the player's radius
    static PULL_STRENGTH = 100;   // Pull is 100x stronger within the zone
    static MAX_SATELLITES = 10; // Maximum number of satellites
    static FOOD_COLOR = 0xFFFF00; // Yellow base color

    constructor(position, velocity, mass, color, radius = null) {
        super(position, velocity, mass, color, radius);
        this.ejectionCooldown = 0;
        this.ejectedMasses = [];
        this.satellites = [];
        this.global_satellite_rotation = 0;
        this.showPullZone = Constants.SHOW_PULL_ZONE;
        this.updateRadius();
        this.mass_ejection_cooldown = 0;
        this.active = true;
        this.SPEED = 200

        // Ensure player has initial velocity at constant speed
        if (this.velocity.length() === 0) {
            // Give random direction if no initial velocity provided
            const angle = Math.random() * Math.PI * 2;
            this.velocity.x = Math.cos(angle);
            this.velocity.y = Math.sin(angle);
            this.velocity.normalize();
            this.velocity.scale(200);
        } else if (this.velocity.length() > 0) {
            // Normalize any provided velocity to constant speed
            this.velocity.normalize();
            this.velocity.scale(200);
        }

        // Add fire beam properties
        this.beam = null;
        this.beamTimer = 0;
        this.fireCooldown = 0;

        // Add battle royale properties
        this.frags = 0;
        this.alive = true;

        // New properties for advanced fire cooldown
        this.shotsRemaining = Constants.MAX_CONSECUTIVE_SHOTS;  // Track available shots
        this.baseShotCooldown = Constants.BEAM_DURATION + 0.1;  // Reduced from +0.2
        this.inReloadPeriod = false;                           // Whether in longer reload period

        // Dynamic multiplier properties
        this.currentMultiplier = Constants.SHOT_COOLDOWN_MULTIPLIER; // Current multiplier value
        this.multiplierTimer = 0;                              // Timer for multiplier reset

        // Add properties needed for sprite handling
        this.sprite = null;
        this.satelliteSprites = []; // Store satellite sprites for this player

        // Track if player is accelerating
        this.accelerating = false;

        // ENHANCED SHOOTING: Override default cooldown values
        this.shotsRemaining = Constants.MAX_CONSECUTIVE_SHOTS * 2; // Double available shots

        // Add rapid fire properties
        this.RAPID_FIRE = true; // Enable rapid fire mode

        // Store original constants for reference
        this._originalMaxShots = Constants.MAX_CONSECUTIVE_SHOTS;

        // Override reload mechanics for faster recovery
        this.handleReloadComplete = function() {
            // Restore twice as many shots and 50% faster reload
            this.shotsRemaining = this._originalMaxShots * 2;
            this.inReloadPeriod = false;
        }
    }

    updateRadius() {
        // Update radius based on current mass (using cube root for volume scaling)
        this.radius = Math.max(this.K * Math.pow(this.mass, 1/3), 1);
    }

    ejectMass(direction, dt, planets) {
        // Cooldown handling (0.1 seconds between ejections for a chain effect)
        const COOLDOWN_TIME = 0.1; // Reduced from typical 0.5
        this.ejectionCooldown -= dt;

        // Check if player can eject mass
        if (this.ejectionCooldown <= 0 && this.mass > Player.MIN_MASS * 1.5) {
            // Reset cooldown
            this.ejectionCooldown = COOLDOWN_TIME;

            // Calculate ejection mass (smaller chunks for chain effect)
            const ejectionMass = this.mass * 0.01; // 1% of current mass

            // Update player mass
            this.mass -= ejectionMass;
            this.updateRadius();

            // Normalize direction
            const dirLength = Math.sqrt(direction.x * direction.x + direction.y * direction.y);
            if (dirLength === 0) return null;

            direction.x /= dirLength;
            direction.y /= dirLength;

            // Calculate ejection velocity (faster for better chain effect)
            const ejectionSpeed = 3000; // Increased from typical 200
            const ejectionVelocity = new Phaser.Math.Vector2(
                this.velocity.x - direction.x * ejectionSpeed,
                this.velocity.y - direction.y * ejectionSpeed
            );

            // Calculate ejection position (start slightly away from player)
            const ejectionPosition = new Phaser.Math.Vector2(
                this.position.x - direction.x * (this.radius + 5),
                this.position.y - direction.y * (this.radius + 5)
            );

            // Create food object with random color variation
            const food = new Food(
                ejectionPosition,
                ejectionVelocity,
                ejectionMass,
                Player.FOOD_COLOR
            );

            // Add to ejected masses list for tracking
            this.ejectedMasses.push(food);
            if (this.ejectedMasses.length > 10) {
                this.ejectedMasses.shift();
            }

            // Change velocity direction only, maintain constant speed
            this.velocity.x += direction.x * ejectionSpeed * ejectionMass / this.mass;
            this.velocity.y += direction.y * ejectionSpeed * ejectionMass / this.mass;

            // Normalize and scale to maintain constant speed
            if (this.velocity.length() > 0) {
                this.velocity.normalize();
                this.velocity.scale(Constants.PLAYER_SPEED);
            }

            return food;
        }

        return null;
    }

    update(dt, planets) {
        // Update beam timer
        if (this.beamTimer > 0) {
            this.beamTimer -= dt;

            // Check beam collisions (only if we have access to gameScene)
            if (window.gameScene) {
                this.checkBeamCollisions(window.gameScene);
            }

            if (this.beamTimer <= 0) {
                this.beamTimer = 0;
                // Create food at end of beam only if beam still exists
                if (this.beam && this.beam.endPoint) {
                    this.createFoodFromBeam();
                }
                this.beam = null;
            }
        }

        // Rest of update method remains the same
        // Update fire cooldown
        if (this.fireCooldown > 0) {
            this.fireCooldown -= dt;
            if (this.fireCooldown <= 0) {
                // If we finished a regular shot cooldown but still in reload period
                if (this.inReloadPeriod) {
                    // Check if reload period is over
                    if (this.shotsRemaining <= 0) {
                        // Reset shots and exit reload period
                        this.shotsRemaining = Constants.MAX_CONSECUTIVE_SHOTS;
                        this.inReloadPeriod = false;
                    }
                }

                // Clear cooldown when it reaches zero
                this.fireCooldown = 0;
            }
        }

        // Calculate gravitational acceleration from planets
        const acceleration = computeAcceleration(this.position, planets);

        this.moveForward(dt)

        // Update shield
        this.updateShield(dt);

        // Update multiplier timer and reset if needed
        this.multiplierTimer += dt;
        if (this.multiplierTimer >= Constants.MULTIPLIER_COOLDOWN) {
            // Reset multiplier to initial value every MULTIPLIER_COOLDOWN seconds
            this.currentMultiplier = Constants.SHOT_COOLDOWN_MULTIPLIER;
            this.multiplierTimer = 0;
        } else {
            // Gradually decrease multiplier between resets
            this.currentMultiplier = Math.max(1.0, this.currentMultiplier - (0.01 * dt));
        }

        // Update sprite visuals
        this.updateSpriteVisuals();

        // Update food attraction
        if (window.gameScene && window.gameScene.foodList) {
            this.updateFoodAttraction(window.gameScene.foodList, dt);
        }

        // --- ADDED: Shield Collision Checks ---
        if (this.alive && window.gameScene) {
            // Check against human player (if this instance is not the human player)
            if (window.gameScene.player && window.gameScene.player !== this && window.gameScene.player.alive) {
                this.checkShieldCollision(window.gameScene.player);
            }

            // Check against other AI players
            if (window.gameScene.aiPlayers) {
                for (const otherAI of window.gameScene.aiPlayers) {
                    if (otherAI && otherAI !== this && otherAI.alive) {
                        this.checkShieldCollision(otherAI);
                    }
                }
            }
        }
        // --- END ADDED: Shield Collision Checks ---
    }

    updateShield(dt) {
        // Create shield if it doesn't exist
        if (!this.shield) {
            this.shield = new Shield(this);
        }

        // Update shield position and rotation
        if (this.shield) {
            this.shield.update(dt);
        }
    }

    handleWrapping() {
        // Implement edge wrapping directly instead of calling super
        // Wrap x-coordinate around the screen edges
        if (this.position.x < 0) {
            this.position.x += Constants.WIDTH;
        } else if (this.position.x > Constants.WIDTH) {
            this.position.x -= Constants.WIDTH;
        }

        // Wrap y-coordinate around the screen edges
        if (this.position.y < 0) {
            this.position.y += Constants.HEIGHT;
        } else if (this.position.y > Constants.HEIGHT) {
            this.position.y -= Constants.HEIGHT;
        }
    }

    drawPullZone(graphics, animationTime) {
        if (this.showPullZone) {
            // Draw pull zone - use a more visible, animated style
            const pulseValue = 0.5 + 0.3 * Math.sin(animationTime * 4);
            graphics.lineStyle(2, 0xffffff, pulseValue);
            graphics.strokeCircle(
                this.position.x,
                this.position.y,
                Player.PULL_ZONE_FACTOR * this.radius / Constants.SCALE_FACTOR
            );
        }
    }

    /**
     * Checks if the player's active beam hits a target player's body.
     * @param {Player} target - The player to check collision against.
     * @returns {boolean} True if the beam hits the target's body, false otherwise.
     */
    checkBeamHit(target) {
        // Add checks for target.position and this.beam.endPoint
        if (!target || !target.position || !target.alive || this.beamTimer <= 0 || !this.beam || !this.beam.endPoint) {
            return false; // No target, target missing position, target dead, beam inactive, or beam object/endpoint missing
        }

        // Beam is a line segment from this.position to beam.end
        const beamStart = this.position;
        const beamEnd = this.beam.endPoint; // Use endPoint consistently

        // Target is a circle at target.position with radius target.radius
        const targetCenter = target.position;
        const targetRadius = target.radius;

        // Use Phaser's geometry intersection check: Line vs Circle
        const line = new Phaser.Geom.Line(beamStart.x, beamStart.y, beamEnd.x, beamEnd.y);
        const circle = new Phaser.Geom.Circle(targetCenter.x, targetCenter.y, targetRadius);

        return Phaser.Geom.Intersects.LineToCircle(line, circle);
    }

    drawShield(graphics) {
        if (this.shield && this.alive) {
            this.shield.draw(graphics);
        }
    }

    applyPullForce(food, dt) {
        // Calculate vector from food to player
        let dx = this.position.x - food.position.x;
        let dy = this.position.y - food.position.y;

        // Handle toroidal wrapping - find the shortest distance
        if (Math.abs(dx) > Constants.WIDTH / 2) {
            dx = dx > 0 ? dx - Constants.WIDTH : dx + Constants.WIDTH;
        }
        if (Math.abs(dy) > Constants.HEIGHT / 2) {
            dy = dy > 0 ? dy - Constants.HEIGHT : dy + Constants.HEIGHT;
        }

        // Calculate distance
        const distance = Math.sqrt(dx * dx + dy * dy);
        const pullZoneRadius = this.radius * Player.PULL_ZONE_FACTOR;

        // Apply pull force if within pull zone
        if (distance < pullZoneRadius && distance > 0) {
            // Calculate pull strength - increases as food gets closer to player
            const pullFactor = Player.PULL_STRENGTH * (1 - distance / pullZoneRadius);

            // Add to food's velocity toward player
            const directionX = dx / distance;
            const directionY = dy / distance;

            // Scale force by dt for time-based physics
            const pullForce = 100 * pullFactor * dt; // Base force of 100 units

            food.velocity.x += directionX * pullForce;
            food.velocity.y += directionY * pullForce;
        }
    }

    collectFood(foodList, foodSprites, scene) {
        const initialMass = this.mass;

        for (let i = foodList.length - 1; i >= 0; i--) {
            const food = foodList[i];

            // Skip invalid food objects
            if (!food || !food.position) continue;

            // VALIDATION: Check for NaN position in food
            if (isNaN(food.position.x) || isNaN(food.position.y)) {
                // Remove invalid food from scene and list
                if (food.sprite) {
                    food.sprite.destroy();
                }
                foodList.splice(i, 1); // Remove from list
                console.warn("Invalid food position detected:", food.position);
                continue;
            }

            // MISSING CODE: Calculate distance between player and food with wrapping
            const dx = this.position.x - food.position.x;
            const dy = this.position.y - food.position.y;

            // Handle wrapping - find shortest distance in wrapped world
            const wrappedDx = Math.min(
                Math.abs(dx),
                Math.abs(dx - Constants.WIDTH),
                Math.abs(dx + Constants.WIDTH)
            );

            const wrappedDy = Math.min(
                Math.abs(dy),
                Math.abs(dy - Constants.HEIGHT),
                Math.abs(dy + Constants.HEIGHT)
            );

            const distance = Math.sqrt(wrappedDx * wrappedDx + wrappedDy * wrappedDy);

            // Now this condition will work properly
            if (distance < this.radius) {
                // IMPORTANT: Validate food mass before adding
                if (!isNaN(food.mass) && isFinite(food.mass) && food.mass > 0 && food.mass < 100) {
                    const MAX_PLAYER_MASS = 1000; // Reasonable upper limit

                    // Add food mass with upper limit
                    this.mass = Math.min(this.mass + food.mass, MAX_PLAYER_MASS);

                    // Update radius
                    this.updateRadius();
                } else {
                    console.warn("Invalid food mass detected:", food.mass);
                }

                // Remove the food regardless of mass validity
                if (food.sprite) {
                    food.sprite.destroy();
                }
                foodList.splice(i, 1);
            }
        }
    }

    // Fire a beam in the opposite direction the player is facing
    fireBeam(gameScene) {

        if (this.fireCooldown > 0 || this.beamTimer > 0 || this.mass < Constants.FIRE_COST || !this.alive) {
            return false;
        }

        // Check if we need to enter reload period
        if (this.shotsRemaining <= 0) {
            this.inReloadPeriod = true;
            this.fireCooldown = Constants.RELOAD_COOLDOWN;
            return false;
        }

        // Consume mass
        this.mass -= Constants.FIRE_COST;
        this.updateRadius();

        // Get player sprite rotation (need to pass from game scene)
        let playerAngle;

        // If game scene has playerSprite, use its rotation
        if (this.sprite) {
            // In Phaser, rotation 0 means pointing right, not up
            // We want to keep the sprite's actual rotation to match its visual direction
            playerAngle = this.sprite.rotation;
        } else if (this.velocity.length() > 0) {
            // Fallback to velocity direction if sprite not available
            playerAngle = Math.atan2(this.velocity.y, this.velocity.x);
        } else {
            // Random direction if no velocity and no sprite
            playerAngle = Math.random() * Math.PI * 2;
        }

        // Add 180 degrees to get the correct direction
        playerAngle += Math.PI;

        // Calculate direction vector based on angle
        const direction = new Phaser.Math.Vector2(
            Math.cos(playerAngle - Math.PI/2), // Adjust by -π/2 since sprite's nose is at rotation 0
            Math.sin(playerAngle - Math.PI/2)
        );

        // Calculate nose position (at the edge of the player in the firing direction)
        const nosePosition = new Phaser.Math.Vector2(
            this.position.x + direction.x * this.radius,
            this.position.y + direction.y * this.radius
        );

        // Rest of the method stays the same
        const beamLength = this.radius * Constants.BEAM_LENGTH_FACTOR;
        const endX = nosePosition.x + direction.x * beamLength;
        const endY = nosePosition.y + direction.y * beamLength;

        this.beam = {
            startPoint: new Phaser.Math.Vector2(nosePosition.x, nosePosition.y),
            endPoint: new Phaser.Math.Vector2(nosePosition.x, nosePosition.y), // Start at nose position
            direction: direction,
            length: beamLength,
            color: this.color,
            initialLength: beamLength,
            alpha: 1.0,
            fixedStartPoint: new Phaser.Math.Vector2(nosePosition.x, nosePosition.y),
            targetEndPoint: new Phaser.Math.Vector2(endX, endY) // Store the target end point
        };

        this.beamTimer = Constants.BEAM_DURATION;

        // Calculate progressive cooldown based on shots fired, using the current dynamic multiplier
        const shotsAlreadyFired = this._originalMaxShots - this.shotsRemaining;
        const reducedMultiplier = this.RAPID_FIRE ?
            Math.sqrt(this.currentMultiplier) : // Square root for much slower growth
            this.currentMultiplier;

        this.fireCooldown = this.baseShotCooldown * Math.pow(reducedMultiplier, shotsAlreadyFired / 2);

        // Decrement shots remaining
        this.shotsRemaining--;

        // If this was our last shot, set up for reload period on next attempt
        if (this.shotsRemaining <= 0) {
            this.inReloadPeriod = true;
        }

        return true;
    }

    // Create food at the end of the beam
    createFoodFromBeam() {
        if (!this.beam || !this.beam.endPoint) return null;

        // Create food at end of beam
        const position = new Phaser.Math.Vector2(this.beam.endPoint.x, this.beam.endPoint.y);

        // Small random velocity
        const angle = Math.random() * Math.PI * 2;
        const speed = 50 / Constants.SCALE_FACTOR;
        const velocity = new Phaser.Math.Vector2(
            Math.cos(angle) * speed,
            Math.sin(angle) * speed
        );

        // Convert the beam energy to food mass
        const foodMass = Constants.FIRE_COST * 0.8; // 80% efficiency

        // Create food with same color as player's beam
        const food = new Food(position, velocity, foodMass, this.color);

        // Add to global food list via game scene
        if (window.gameScene && window.gameScene.foodList) {
            window.gameScene.foodList.push(food);
        }

        return food;
    }

    // Draw beam if active
    drawBeam(graphics, animationTime) {
        if (!this.beam || this.beamTimer <= 0) return;

        // Calculate progress (0 to 1, where 0 is just started and 1 is ending)
        const progress = this.beamTimer / Constants.BEAM_DURATION;

        // REVERSED GROWTH: Beam grows from start to end point
        // When progress = 1 (just started), beam length is 0% of full length
        // When progress = 0 (about to end), beam length is 100% of full length
        const growthFactor = 1 - progress;  // Invert progress for growth

        // Calculate current end point based on growth
        const currentEndX = this.beam.startPoint.x + this.beam.direction.x * this.beam.initialLength * growthFactor;
        const currentEndY = this.beam.startPoint.y + this.beam.direction.y * this.beam.initialLength * growthFactor;

        // Calculate beam alpha and width based on time remaining
        const alpha = progress * 0.9 + 0.1; // Keep minimum alpha so beam is always visible
        const baseWidth = 2 + 4 * (1 - progress); // Thicker at start, thinner at end

        // Draw main beam (brighter core)
        graphics.lineStyle(baseWidth, this.beam.color, alpha);
        graphics.beginPath();
        graphics.moveTo(this.beam.startPoint.x, this.beam.startPoint.y);
        graphics.lineTo(currentEndX, currentEndY);
        graphics.strokePath();

        // Draw outer glow effect (thicker but more transparent)
        graphics.lineStyle(baseWidth + 2, this.beam.color, alpha * 0.4);
        graphics.beginPath();
        graphics.moveTo(this.beam.startPoint.x, this.beam.startPoint.y);
        graphics.lineTo(currentEndX, currentEndY);
        graphics.strokePath();

        // Update beam's current end point for collision detection
        this.beam.endPoint.x = currentEndX;
        this.beam.endPoint.y = currentEndY;

        // Optional pulse effect using animationTime if provided
        if (animationTime !== undefined) {
            const pulseSize = 5 + Math.sin(animationTime * 10) * 2;
            const pulseAlpha = 0.3;

            // Draw pulsing point at beam start
            graphics.fillStyle(this.beam.color, pulseAlpha);
            graphics.fillCircle(this.beam.startPoint.x, this.beam.startPoint.y, pulseSize);
        }
    }

    // Check beam collisions with planets and players
    checkBeamCollisions(gameScene) {
        if (!this.beam || this.beamTimer <= 0) return false;

        // Get line segments of the beam
        const line = {
            x1: this.beam.startPoint.x,
            y1: this.beam.startPoint.y,
            x2: this.beam.endPoint.x,
            y2: this.beam.endPoint.y
        };

        // Check collision with planets
        for (const planet of gameScene.planets) {
            // Check if line intersects with planet circle
            if (this.lineCircleIntersection(line, planet)) {
                this.handleBeamPlanetCollision(planet, gameScene);
                return true;
            }
        }

        // Check collision with human player (if not self)
        if (gameScene.player !== this && this.lineCircleIntersection(line, gameScene.player)) {
            // Check if beam is blocked by player's shield
            if (gameScene.player.shield && gameScene.player.shield.isActive() && gameScene.player.shield.checkBeamCollision(this.beam)) {
                // Pass scene and position to takeDamage, which now handles the effect
                const hitPosition = this.beam.endPoint;
                // <<< PASS 'this' (the current player instance) as the attacker
                gameScene.player.shield.takeDamage(gameScene, hitPosition, this);
                this.beamTimer = 0; // Terminate beam
                this.createShieldHitEffect(gameScene, hitPosition);
                if (gameScene.player.shield.currentStrength > 10) {
                    return true;
                }
            }

            // No shield block - player takes damage
            gameScene.player.alive = false;
            gameScene.displayGameOver("You were eliminated by an AI player!");
            return true;
        }

        // Check collision with AI players
        for (let i = 0; i < gameScene.aiPlayers.length; i++) {
            const aiPlayer = gameScene.aiPlayers[i];

            // Skip if checking against self or already dead players
            if (aiPlayer === this || !aiPlayer.alive) continue;

            if (this.lineCircleIntersection(line, aiPlayer)) {
                // Check if beam is blocked by AI player's shield
                if (aiPlayer.shield && aiPlayer.shield.isActive() && aiPlayer.shield.checkBeamCollision(this.beam)) {
                    // Pass scene and position to takeDamage, which now handles the effect
                    const hitPosition = this.beam.endPoint;
                    // <<< PASS 'this' (the current player instance) as the attacker
                    aiPlayer.shield.takeDamage(gameScene, hitPosition, this);

                    this.beamTimer = 0; // Terminate beam
                    if (aiPlayer.shield.currentStrength >= 0) {
                        return true;
                    }
                }

                // No shield block - AI player takes damage
                aiPlayer.die("shield is down", this);
                gameScene.respawnAIPlayer(i, this);
                return true;
            }
        }

        return false;
    }

    // Add shield hit effect method
    createShieldHitEffect(scene, position) {
        // Create shield hit particle effect
        // --- ADD VALIDATION ---
        if (!scene || typeof scene.add !== 'object' || typeof scene.add.particles !== 'function') {
            console.error("createShieldHitEffect: Invalid scene object received.", scene);
            return; // Exit if scene is invalid
        }
        // --- END VALIDATION ---
        const particles = scene.add.particles('particle');
        const emitter = particles.createEmitter({
            x: position.x,
            y: position.y,
            speed: { min: 30, max: 80 },
            angle: { min: 0, max: 360 },
            scale: { start: 0.5, end: 0 },
            blendMode: 'ADD',
            lifespan: 300,
            tint: 0x88FFFF,
            quantity: 10
        });

        // Auto-destroy after effect completes
        scene.time.delayedCall(300, () => {
            emitter.stop();
            scene.time.delayedCall(300, () => {
                particles.destroy();
            });
        });
    }

    handleBeamPlayerCollision(scene, beam, shooter) {
        // Skip if player is not alive or beam is already collided
        if (!this.alive || beam.collided) return;

        // Check if any satellite blocks the beam
        if (this.checkSatelliteBeamBlock(beam, scene)) {
            // Beam was blocked by satellite - don't damage player
            beam.collided = true;
            return;
        }

        // Player was hit - handle player damage
        beam.collided = true; // Mark beam as collided

        // Kill player and credit the shooter
        this.die("Shot by beam", shooter);

        // Increment shooter's frag count if provided
        if (shooter) {
            shooter.frags = (shooter.frags || 0) + 1;
        }

        // Create death effect
        this.createDeathEffect(scene);

        return true; // Return true to indicate a successful hit
    }

    createDeathEffect(scene) {
        if (!scene || !scene.add) return;

        // Create explosion particle effect
        const particles = scene.add.particles('particle');
        const emitter = particles.createEmitter({
            x: this.position.x,
            y: this.position.y,
            speed: { min: 50, max: 200 },
            angle: { min: 0, max: 360 },
            scale: { start: 1.0, end: 0 },
            blendMode: 'ADD',
            lifespan: 500,
            tint: this.color || 0xFFFFFF,
            quantity: 30
        });

        // Auto-destroy after effect completes
        scene.time.delayedCall(500, () => {
            emitter.stop();
            scene.time.delayedCall(500, () => {
                particles.destroy();
            });
        });
    }

    // Helper method to detect line-circle intersection
    lineCircleIntersection(line, circle) {
        // Line equation: (x2-x1)(y-y1) - (y2-y1)(x-x1) = 0
        // Circle equation: (x-cx)² + (y-cy)² = r²

        const x1 = line.x1;
        const y1 = line.y1;
        const x2 = line.x2;
        const y2 = line.y2;

        const cx = circle.position.x;
        const cy = circle.position.y;
        const r = circle.radius;

        // Vector from point 1 to point 2
        const dx = x2 - x1;
        const dy = y2 - y1;

        // Vector from point 1 to circle center
        const pcx = cx - x1;
        const pcy = cy - y1;

        // Project circle center onto line
        const lineLength = Math.sqrt(dx * dx + dy * dy);
        const dot = (pcx * dx + pcy * dy) / lineLength;

        // Find closest point on line to circle center
        let closest = { x: x1 + dx * dot / lineLength, y: y1 + dy * dot / lineLength };

        // Check if closest point is on the line segment
        const onSegment = dot >= 0 && dot <= lineLength;

        if (!onSegment) {
            // If closest point is not on segment, check endpoints
            const dist1 = Math.sqrt(Math.pow(cx - x1, 2) + Math.pow(cy - y1, 2));
            const dist2 = Math.sqrt(Math.pow(cx - x2, 2) + Math.pow(cy - y2, 2));

            if (dist1 <= r || dist2 <= r) return true;
            return false;
        }

        // Check if closest point is within circle radius
        const distance = Math.sqrt(Math.pow(closest.x - cx, 2) + Math.pow(closest.y - cy, 2));
        return distance <= r;
    }

    // Handle planet collision by creating food particles
    handleBeamPlanetCollision(planet, gameScene) {
        // Calculate collision point (approximate)
        // Find intersection point with planet's surface
        const dx = this.beam.endPoint.x - this.beam.startPoint.x;
        const dy = this.beam.endPoint.y - this.beam.startPoint.y;
        const length = Math.sqrt(dx * dx + dy * dy);

        // Normalize
        const unitX = dx / length;
        const unitY = dy / length;

        // Calculate planet direction from beam start
        const planetDx = planet.position.x - this.beam.startPoint.x;
        const planetDy = planet.position.y - this.beam.startPoint.y;

        // Project onto beam direction
        const projection = planetDx * unitX + planetDy * unitY;

        // Calculate projected point
        const projectedX = this.beam.startPoint.x + unitX * projection;
        const projectedY = this.beam.startPoint.y + unitY * projection;

        // Calculate vector from projected point to planet center
        const toPlanetX = planet.position.x - projectedX;
        const toPlanetY = planet.position.y - projectedY;
        const distToPlanet = Math.sqrt(toPlanetX * toPlanetX + toPlanetY * toPlanetY);

        // Normalize vector and scale to planet radius
        const normalizedX = toPlanetX / distToPlanet;
        const normalizedY = toPlanetY / distToPlanet;

        // Move from planet center to surface
        const collisionX = planet.position.x - normalizedX * planet.radius;
        const collisionY = planet.position.y - normalizedY * planet.radius;

        this.burstIntoFood(gameScene, collisionX, collisionY);

        // Terminate the beam by setting timer to 0
        this.beamTimer = 0;
        this.beam = null;
    }

    burstIntoFood(gameScene, collisionX, collisionY) {
        const foodCount = 20;  // Constant number of food particles

        for (let i = 0; i < foodCount; i++) {
            // Create burst direction with randomized angle
            const angle = Math.random() * Math.PI * 2;
            const burstSpeed = 50; // + Math.random() * 100;

            // Random position offset from player center
            const offsetDistance = Math.random() * this.radius * 2.8;
            const offsetAngle = Math.random() * Math.PI * 2;

            const startPosition = new Phaser.Math.Vector2(
                collisionX + Math.cos(offsetAngle) * offsetDistance,
                collisionY + Math.sin(offsetAngle) * offsetDistance
            );

            const velocity = new Phaser.Math.Vector2(
                Math.cos(angle) * burstSpeed,
                Math.sin(angle) * burstSpeed
            );

            // Create food with player's color
            const foodMass = 1 + Math.random() * 4;  // Random mass between 1-5
            const food = new Food(startPosition, velocity, foodMass, Colors.YELLOW);
            gameScene.foodList.push(food);
        }
    }

    die(reason = null, killer = null) {
        if (!this.alive) return;
        this.alive = false;
        killer.frags = (killer.frags || 0) + 1; // Increment killer's frag count

        // Record death information
        this.deathReason = reason;
        this.lastHitBy = killer;

        // Clean up satellite sprites when player dies
        if (this.satelliteSprites) {
            this.satelliteSprites.forEach(sat => {
                if (sat.sprite) {
                    sat.sprite.destroy();
                }
            });
            this.satelliteSprites = [];
        }

        // ADD THIS: Clean up player sprite
        if (this.sprite) {
            this.sprite.destroy();
            this.sprite = null;
        }

        // Empty satellites array
        this.satellites = [];

        this.burstIntoFood(window.gameScene, this.position.x, this.position.y);
    }

    moveForward(dt) {
        if (!this.sprite) return;

        // Get the current sprite rotation
        const playerAngle = this.sprite.rotation;

        // Calculate direction vector from sprite rotation
        const directionX = Math.cos(playerAngle + Math.PI/2);
        const directionY = Math.sin(playerAngle + Math.PI/2);

        // Apply fixed speed in the facing direction
        // Speed is now independent of mass
        const speed = this.accelerating ? Constants.RUN_SPEED : Constants.BASE_SPEED;

        // Set velocity directly based on direction and fixed speed
        this.velocity.x = directionX * speed;
        this.velocity.y = directionY * speed;

        // Update position
        this.position.x += this.velocity.x * dt;
        this.position.y += this.velocity.y * dt;

        // Handle wrapping around screen edges
        this.handleWrapping();
    }

    // Property already defined in constructor
    // this.sprite = null;

    // Method to assign sprite to this player
    setSprite(sprite) {
        this.sprite = sprite;
        this.updateSpriteVisuals();
    }

    // Update sprite position, rotation, and scale
    updateSpriteVisuals() {
        if (!this.sprite) return;

        // Update position
        this.sprite.x = this.position.x;
        this.sprite.y = this.position.y;

        // FIXED: Update size based on DIAMETER (radius * 2)
        const size = this.radius * 2;  // Changed from radius to radius*2
        this.sprite.setDisplaySize(size, size);

        // If this is an AI player with color, use the color
        if (this.color !== undefined) {
            this.sprite.setTint(this.color);
        }

        // Apply color based on acceleration state
        if (this.accelerating) {
            // Apply brighter tint or effect when accelerating
            this.sprite.setTint(this.color !== undefined ? this.color : 0xFFFFFF);
            this.sprite.setAlpha(1.2); // Slight glow effect
        } else {
            // Normal appearance
            if (this.color !== undefined) {
                this.sprite.setTint(this.color);
            }
            this.sprite.setAlpha(1.0);
        }
    }

    // Add method for AIPlayer to face a target without changing movement
    facePoint(targetPosition) {
        if (!this.sprite) return;

        // Calculate direction to point
        let dx = targetPosition.x - this.position.x;
        let dy = targetPosition.y - this.position.y;

        // Handle wrapping
        if (Math.abs(dx) > Constants.WIDTH / 2) {
            dx = dx > 0 ? dx - Constants.WIDTH : dx + Constants.WIDTH;
        }
        if (Math.abs(dy) > Constants.HEIGHT / 2) {
            dy = dy > 0 ? dy - Constants.HEIGHT : dy + Constants.HEIGHT;
        }

        // Set sprite rotation to face the target
        this.sprite.rotation = Math.atan2(dy, dx) + Math.PI/2;
    }

    // Add to CelestialBody class or Player class
    computeAcceleration(planets) {
        let ax = 0, ay = 0;

        // Iterate through each planet
        for (const planet of planets) {
            // Calculate vector from this object to planet
            const dx = planet.position.x - this.position.x;
            const dy = planet.position.y - this.position.y;

            // Calculate distance squared
            const distSquared = dx * dx + dy * dy;
            const dist = Math.sqrt(distSquared);

            // Define gravity influence range (5x planet radius)
            const gravityRange = planet.radius * 5;

            // Only apply gravity if within range
            if (dist <= gravityRange) {
                // Skip if extremely close to avoid division by tiny numbers
                if (dist < planet.radius * 0.2) continue;

                // Calculate unit direction vector
                const nx = dx / dist;
                const ny = dy / dist;

                // Calculate gravity strength with inverse square law
                const gravityStrength = Constants.Gravity * planet.mass / distSquared;

                // Add contribution to acceleration vector
                ax += nx * gravityStrength;
                ay += ny * gravityStrength;
            }
        }

        return { x: ax, y: ay };
    }

    // Add to Player class
    updateFoodAttraction(foodList, dt) {
        if (!foodList || !this.alive) return;

        // Iterate through all food items
        for (const food of foodList) {
            // Skip invalid food objects
            if (!food || !food.position) continue;

            // Apply pull force to each food item
            this.applyPullForce(food, dt);
        }
    }


    // Add to handleBeamPlayerCollision method in main.js
    handleBeamPlayerCollision(scene, beam) {
        // Skip if player is not alive or beam is already collided
        if (!this.alive || beam.collided) return;

        // Check if any satellite blocks the beam
        if (this.checkSatelliteBeamBlock(beam, scene)) {
            // Beam was blocked by satellite - don't damage player
            beam.collided = true;
            return;
        }

        // Continue with existing collision logic for player...
        // Existing player hit code...
    }

    // Add this new method to the GameScene class
    checkSatelliteBeamBlock(beam, scene) {
        // Skip if player has no satellites
        if (!this.satellites || this.satellites.length === 0) return false;

        // Store beam start and end points
        const beamStart = { x: beam.startX, y: beam.startY };
        const beamEnd = { x: beam.endX, y: beam.endY };

        // Check each satellite for collision with beam
        for (let i = 0; i < this.satellites.length; i++) {
            const satellite = this.satellites[i];

            // Calculate distance from satellite to beam
            const distance = this.pointLineDistance(
                satellite.position,
                beamStart,
                beamEnd
            );

            // Get satellite radius (with hitbox multiplier)
            const satelliteRadius = satellite.radius * Constants.SATELLITE_HITBOX;

            // Check if beam hit satellite
            if (distance <= satelliteRadius) {
                // Beam hit a satellite!
                this.handleSatelliteBeamBlock(scene, beam, satellite, i);
                return true;
            }
        }

        return false; // No satellites blocked the beam
    }

    // Add method to handle satellite beam block
    handleSatelliteBeamBlock(scene, beam, satellite, satelliteIndex) {
        // Only block if random chance succeeds
        if (Math.random() > Constants.SATELLITE_BLOCK_CHANCE) return false;

        // Stop beam from continuing
        beam.collided = true;

        // Create visual effect
        this.createSatelliteBlockEffect(scene, satellite.position);

        // Destroy satellite if enabled
        if (Constants.SATELLITE_DESTRUCTION) {
            // Remove from player's satellites array
            this.satellites.splice(satelliteIndex, 1);

            // Destroy sprite if it exists
            if (satellite.sprite) {
                satellite.sprite.destroy();
            }

            // Update this radius to reflect lost satellite mass
            this.updateRadius();
        }

        return true;
    }

    // Helper method for calculating distance from point to line segment
    pointLineDistance(point, lineStart, lineEnd) {
        // Vector from start to end
        const dx = lineEnd.x - lineStart.x;
        const dy = lineEnd.y - lineStart.y;

        // Length of line squared
        const lenSq = dx*dx + dy*dy;

        // If line is a point, return distance to that point
        if (lenSq === 0) {
            const distX = point.x - lineStart.x;
            const distY = point.y - lineStart.y;
            return Math.sqrt(distX*distX + distY*distY);
        }

        // Calculate projection of point onto line
        const t = ((point.x - lineStart.x) * dx + (point.y - lineStart.y) * dy) / lenSq;

        // If outside line segment, use distance to nearest endpoint
        if (t < 0) {
            // Beyond start point
            const distX = point.x - lineStart.x;
            const distY = point.y - lineStart.y;
            return Math.sqrt(distX*distX + distY*distY);
        }
        if (t > 1) {
            // Beyond end point
            const distX = point.x - lineEnd.x;
            const distY = point.y - lineEnd.y;
            return Math.sqrt(distX*distX + distY*distY);
        }

        // Perpendicular distance to line
        const projX = lineStart.x + t * dx;
        const projY = lineStart.y + t * dy;
        const distX = point.x - projX;
        const distY = point.y - projY;

        return Math.sqrt(distX*distX + distY*distY);
    }

    // Create visual effect for satellite block
    createSatelliteBlockEffect(scene, position) {
        // Create explosion particle effect
        const particles = scene.add.particles('particle');
        const emitter = particles.createEmitter({
            x: position.x,
            y: position.y,
            speed: { min: 20, max: 100 },
            angle: { min: 0, max: 360 },
            scale: { start: 0.6, end: 0 },
            blendMode: 'ADD',
            lifespan: 300,
            tint: 0x88FFFF,
            quantity: 15
        });

        // Auto-destroy after effect completes
        scene.time.delayedCall(300, () => {
            emitter.stop();
            scene.time.delayedCall(300, () => {
                particles.destroy();
            });
        });
    }

    // Add shield collision detection method to Player class
    checkShieldCollision(otherPlayer) {
        if (!this.alive || !otherPlayer.alive) return false;

        // Get actual shield radii for both players (shield radius = orbitDistance)
        const thisShieldRadius = this.radius * 1.5; // Same calculation as in Shield class
        const otherShieldRadius = otherPlayer.radius * 1.5;

        // Calculate distance between players with toroidal wrapping
        let dx = this.position.x - otherPlayer.position.x;
        let dy = this.position.y - otherPlayer.position.y;

        // Handle toroidal wrapping
        if (Math.abs(dx) > Constants.WIDTH / 2) {
            dx = dx > 0 ? dx - Constants.WIDTH : dx + Constants.WIDTH;
        }
        if (Math.abs(dy) > Constants.HEIGHT / 2) {
            dy = dy > 0 ? dy - Constants.HEIGHT : dy + Constants.HEIGHT;
        }

        // Calculate distance between centers
        const distance = Math.sqrt(dx * dx + dy * dy);

        // Check if shields are colliding
        const minDistance = thisShieldRadius + otherShieldRadius;
        if (distance < minDistance) {
            // Collision detected! Calculate bounce
            this.handleShieldBounce(otherPlayer, dx, dy, distance, minDistance);
            return true;
        }

        return false;
    }

    // Handle the shield bounce physics
    handleShieldBounce(otherPlayer, dx, dy, distance, minDistance) {
`        // Normalize the collision vector (points from otherPlayer to this player)
        const nx = dx / distance;
        const ny = dy / distance;

        // Calculate relative velocity
        const velXDiff = this.velocity.x - otherPlayer.velocity.x;
        const velYDiff = this.velocity.y - otherPlayer.velocity.y;

        // Calculate impulse scalar (how much bounce)
        // A factor > 1 means they bounce off more energetically than they collided
        // A factor of 2 would be a perfectly elastic collision if masses were equal and one was stationary.
        // We use 1.8 for a slightly less than perfectly elastic but still strong bounce.
        const impulse = (velXDiff * nx + velYDiff * ny) * 1.8; 

        // Apply impulse based on mass ratio
        const totalMass = this.mass + otherPlayer.mass;
        const thisRatio = (2 * otherPlayer.mass) / totalMass; // Standard elastic collision formula component
        const otherRatio = (2 * this.mass) / totalMass;   // Standard elastic collision formula component

        // Apply velocity changes (bounce effect)
        // For 'this' player, the impulse is along (-nx, -ny) relative to its original velocity contribution to the impact
        // For 'otherPlayer', the impulse is along (nx, ny) relative to its original velocity contribution
        
        // Simplified impulse application:
        // Each player receives a portion of the total impulse, scaled by the other's mass contribution
        // This ensures momentum conservation.
        const thisPlayerImpulseMagnitude = impulse * (otherPlayer.mass / totalMass);
        const otherPlayerImpulseMagnitude = impulse * (this.mass / totalMass);

        this.velocity.x -= thisPlayerImpulseMagnitude * nx * 1.8; // Apply bounciness factor again or adjust initial impulse
        this.velocity.y -= thisPlayerImpulseMagnitude * ny * 1.8;
        otherPlayer.velocity.x += otherPlayerImpulseMagnitude * nx * 1.8;
        otherPlayer.velocity.y += otherPlayerImpulseMagnitude * ny * 1.8;


        // Push players apart to prevent sticking
        const overlap = minDistance - distance + 1; // Add a small epsilon to ensure separation
        this.position.x += nx * overlap * (otherPlayer.mass / totalMass);
        this.position.y += ny * overlap * (otherPlayer.mass / totalMass);
        otherPlayer.position.x -= nx * overlap * (this.mass / totalMass);
        otherPlayer.position.y -= ny * overlap * (this.mass / totalMass);

        // --- ADDED: Change facing direction ---
        // 'this' player should face in the direction of the bounce (nx, ny)
        if (this.sprite) {
            this.sprite.rotation = Math.atan2(ny, nx) + Math.PI / 2;
        }
        // 'otherPlayer' should face in the direction of its bounce (-nx, -ny)
        if (otherPlayer.sprite) {
            otherPlayer.sprite.rotation = Math.atan2(-ny, -nx) + Math.PI / 2;
        }
        // For AIPlayer, if it uses targetRotation for its state machine,
        // this direct sprite.rotation change will be an immediate override.
        // If AIPlayer has a targetRotation property, you might want to update that too:
        if (typeof this.targetRotation !== 'undefined') {
            this.targetRotation = Math.atan2(ny, nx) + Math.PI / 2;
        }
        if (typeof otherPlayer.targetRotation !== 'undefined') {
            otherPlayer.targetRotation = Math.atan2(-ny, -nx) + Math.PI / 2;
        }
        // --- END ADDED: Change facing direction ---
    }
}