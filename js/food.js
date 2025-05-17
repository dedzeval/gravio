class Food extends CelestialBody {
    static FOOD_MASS_MIN = 0.1;
    static FOOD_MASS_MAX = 0.5;
    static VELOCITY_FACTOR = 3.6; // Controls food speed
    static STAR_ANIMATION_LENGTH = 12; // Number of frames in the star animation
    static STAR_ANIMATION_SPEED = 10; // Speed of the star animation

    constructor(position, velocity, mass = 1, color = Colors.YELLOW) {
        super(position, velocity, mass, color);
        this.animationPhase = Math.random();

        // Add random initial rotation
        this.rotationPhase = Math.random() * Math.PI * 2;
        // Add random rotation speed (positive or negative)
        this.rotationSpeed = (Math.random() * 2 - 1) * 1.5; // Between -1.5 and 1.5 radians per second

        // Slow down initial velocity by factor of 3
        this.velocity.x *= Food.VELOCITY_FACTOR;
        this.velocity.y *= Food.VELOCITY_FACTOR;

        // Ensure food has a minimum visible size
        this.radius = Math.max(this.K * Math.sqrt(this.mass), 3);

        // Store sprite reference
        this.sprite = null;
    }

    // Add static methods for food generation
    static generateAroundPlanet(count, planet) {
        const food = [];

        for (let i = 0; i < count; i++) {
            // Generate position in orbit around planet
            const orbitDistance = planet.radius * 2 + Math.random() * 300;
            const angle = Math.random() * Math.PI * 2;

            const position = new Phaser.Math.Vector2(
                planet.position.x + Math.cos(angle) * orbitDistance,
                planet.position.y + Math.sin(angle) * orbitDistance
            );

            // Generate initial velocity - tangential to orbit but slower by SCALE_FACTOR
            const orbitSpeed = (50 + Math.random() * 50) / Constants.SCALE_FACTOR;
            const tangentX = -Math.sin(angle);
            const tangentY = Math.cos(angle);

            const velocity = new Phaser.Math.Vector2(
                tangentX * orbitSpeed,
                tangentY * orbitSpeed
            );

            // Random mass between 1 and 5
            const mass = 1 + Math.random() * 4;

            // Create food with slightly randomized color
            const food_obj = new Food(position, velocity, mass, Colors.YELLOW);

            food.push(food_obj);
        }

        return food;
    }

    static generateRandom(count) {
        const food = [];

        for (let i = 0; i < count; i++) {
            // Random position across the game area
            const position = new Phaser.Math.Vector2(
                Math.random() * Constants.WIDTH,
                Math.random() * Constants.HEIGHT
            );

            // Random initial velocity - slower by SCALE_FACTOR
            const angle = Math.random() * Math.PI * 2;
            const speed = (20 + Math.random() * 80) / Constants.SCALE_FACTOR;

            const velocity = new Phaser.Math.Vector2(
                Math.cos(angle) * speed,
                Math.sin(angle) * speed
            );

            // Random mass between 1 and 5
            const mass = 1 + Math.random() * 4;

            // Create food with slightly randomized color
            const food_obj = new Food(position, velocity, mass, Colors.YELLOW);

            food.push(food_obj);
        }

        return food;
    }

    update(dt, planets, scene, animationTime) {
        // First apply gravity
        if (planets && planets.length > 0) {
            this.applyGravity(dt, planets);
        }

        // Apply velocity to position
        this.position.x += this.velocity.x * dt;
        this.position.y += this.velocity.y * dt;

        // FIX: Add collision handling with planets
        if (planets && planets.length > 0) {
            this.handlePlanetCollisions(planets);
        }

        // Limit velocity after collision
        this.limitVelocity();

        // Handle wrapping around edges after collision handling
        this.handleWrapping();

        // Then update sprite
        if (scene) {
            this.updateSprite(scene, animationTime);
        }
    }

    handlePlanetCollisions(planets) {
        planets.forEach(planet => {
            const dx = this.position.x - planet.position.x;
            const dy = this.position.y - planet.position.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            const minDistance = planet.radius + this.radius;

            if (distance < minDistance) {
                // SAFETY CHECK: Prevent division by zero
                if (distance < 0.01) {
                    // Food is exactly at planet center - move it out in random direction
                    const randomAngle = Math.random() * Math.PI * 2;
                    this.position.x = planet.position.x + Math.cos(randomAngle) * (minDistance + 1);
                    this.position.y = planet.position.y + Math.sin(randomAngle) * (minDistance + 1);
                    this.velocity.x = Math.cos(randomAngle) * 150; // Increased escape velocity
                    this.velocity.y = Math.sin(randomAngle) * 150;
                    return;
                }

                // Calculate normalized collision normal
                const nx = dx / distance;
                const ny = dy / distance;

                // Calculate velocity component along the normal
                const dot = this.velocity.x * nx + this.velocity.y * ny;

                // Only bounce if moving toward the planet
                if (dot < 0) {
                    // Calculate reflection vector
                    const reflectX = this.velocity.x - 2 * dot * nx;
                    const reflectY = this.velocity.y - 2 * dot * ny;

                    // VALIDATION: Check for NaN before assigning
                    if (!isNaN(reflectX) && !isNaN(reflectY)) {
                        // ENHANCED BOUNCE: Use higher multiplier and add randomness
                        const bounceFactor = 3.5 + Math.random() * 1.5; // Between 3.5-5.0

                        // Apply higher bounce energy
                        this.velocity.x = reflectX * bounceFactor;
                        this.velocity.y = reflectY * bounceFactor;

                        // Ensure minimum bounce velocity
                        const currentSpeed = Math.sqrt(this.velocity.x * this.velocity.x + this.velocity.y * this.velocity.y);
                        const minBounceSpeed = 200 / Constants.SCALE_FACTOR;

                        if (currentSpeed < minBounceSpeed) {
                            const speedRatio = minBounceSpeed / currentSpeed;
                            this.velocity.x *= speedRatio;
                            this.velocity.y *= speedRatio;
                        }
                    } else {
                        // Fallback if NaN detected - use stronger escape velocity
                        const escapeAngle = Math.atan2(ny, nx);
                        const escapeSpeed = 200 + Math.random() * 100; // Higher escape speed
                        this.velocity.x = Math.cos(escapeAngle) * escapeSpeed;
                        this.velocity.y = Math.sin(escapeAngle) * escapeSpeed;
                    }
                }

                // Move food outside the planet to prevent getting stuck
                // Also push further out to prevent immediate re-collision
                const pushOut = minDistance - distance + 5; // More distance to prevent sticking
                this.position.x += nx * pushOut;
                this.position.y += ny * pushOut;

                // Validate mass is still a proper number
                if (isNaN(this.mass) || this.mass <= 0) {
                    this.mass = 1; // Reset to minimum valid mass
                }
            }
        });
    }

    limitVelocity() {
        const maxSpeed = 500 / Constants.SCALE_FACTOR;
        const currentSpeed = Math.sqrt(this.velocity.x * this.velocity.x + this.velocity.y * this.velocity.y);
        if (currentSpeed > maxSpeed) {
            const scaleFactor = maxSpeed / currentSpeed;
            this.velocity.x *= scaleFactor;
            this.velocity.y *= scaleFactor;
        }
    }

    handleWrapping() {
        this.position.x = (this.position.x + Constants.WIDTH) % Constants.WIDTH;
        this.position.y = (this.position.y + Constants.HEIGHT) % Constants.HEIGHT;
    }

    // Add method to assign sprite to this food
    setSprite(sprite) {
        this.sprite = sprite;
    }

    // Fix food sprite color issue
    updateSprite(scene, animationTime) {
        // Create sprite if it doesn't exist
        if (!this.sprite && scene) {
            this.sprite = scene.add.sprite(this.position.x, this.position.y, 'food');
            this.sprite.setOrigin(0.5, 0.5);
        }

        if (this.sprite) {
            // Update position
            this.sprite.x = this.position.x;
            this.sprite.y = this.position.y;

            // Update size with animation effect
            const pulseFactor = 0.9 + 0.1 * Math.sin(this.animationPhase + animationTime * Food.STAR_ANIMATION_SPEED);
            const displaySize = this.radius * 2 * pulseFactor;
            this.sprite.setDisplaySize(displaySize, displaySize);

            // APPLY ROTATION - use both initial phase and continuous rotation
            this.sprite.rotation = this.rotationPhase + (animationTime * this.rotationSpeed);

            // IMPORTANT FIX: Apply color with stronger tint to overcome frame coloring
            this.sprite.setTint(this.color || Colors.YELLOW);

            // Handle star animation frame selection
            const frameIndex = Math.floor(animationTime * Food.STAR_ANIMATION_SPEED + this.animationPhase) % Food.STAR_ANIMATION_LENGTH;
            const frameFormatted = frameIndex.toString().padStart(2, '0');

            // Apply texture frame AFTER tint for better color application
            this.sprite.setTexture(`star_${frameFormatted}`);

            // Force blending mode to ensure color is properly applied
            this.sprite.setBlendMode(Phaser.BlendModes.ADD);
        }
    }

    // Destroy sprite with animation
    destroySprite(scene) {
        if (!this.sprite) return;

        // Add disappearing animation
        if (scene && scene.tweens) {
            scene.tweens.add({
                targets: this.sprite,
                scale: 0,
                alpha: 0,
                duration: 150,
                onComplete: () => {
                    this.sprite.destroy();
                    this.sprite = null;
                }
            });
        } else {
            // Fallback if no scene provided
            this.sprite.destroy();
            this.sprite = null;
        }
    }
}