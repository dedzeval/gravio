class Planet extends CelestialBody {
    static PLANET_MASS = 10000;
    static PLANET_RADIUS = 20;
    static DEFAULT_ORBIT_RADIUS = 250;
    static DEFAULT_ANGULAR_VELOCITY = 0.88;

    constructor(orbit_radius, angular_velocity, mass, color, radius = null) {
        // First initialize with center position
        super(
            new Phaser.Math.Vector2(Constants.center_x, Constants.center_y),
            new Phaser.Math.Vector2(0, 0),
            mass,
            color,
            radius
        );

        // Store orbit parameters
        this.orbit_radius = orbit_radius;
        this.angular_velocity = angular_velocity*1.1;
        this.angle = 0;

        // Reference to the sprite (will be set later)
        this.sprite = null;
    }

    update(dt) {
        // Calculate required angular velocity for a full rotation in 20 seconds
        // With TIME_SCALE of 0.25, we need 2π / (20 * 0.25) = 1.26 rad/s
        const baseAngularVelocity = .5 / Constants.SCALE_FACTOR; // Much faster than 0.1!

        // Use either the passed angular velocity or the calculated base velocity
        const effectiveVelocity = this.angular_velocity > 1 ?
            this.angular_velocity : baseAngularVelocity;

        // Update angle based on angular velocity
        this.angle += effectiveVelocity * dt;
        this.angle %= (2 * Math.PI);

        // Update position based on angle
        this.position.x = Constants.center_x + this.orbit_radius * Math.cos(this.angle);
        this.position.y = Constants.center_y + this.orbit_radius * Math.sin(this.angle);

        // Update sprite position if it exists
        this.updateSprite();
    }

    // Method to update sprite visuals
    updateSprite() {
        if (this.sprite) {
            this.sprite.setPosition(this.position.x, this.position.y);
            this.sprite.setDisplaySize(this.radius * 2, this.radius * 2);
        }
    }

    // Method to create sprite for this planet in the scene
    createSprite(scene) {
        if (!scene) return null;

        // Create sprite using the circular texture
        this.sprite = scene.add.sprite(this.position.x, this.position.y, 'planet_circular');
        this.sprite.setOrigin(0.5, 0.5);

        // Set texture filtering for better quality when scaled
        this.sprite.setInteractive(); // Needed for some Phaser versions to apply texture filtering

        return this.sprite;
    }

    // Check collision with another celestial body
    checkCollision(otherBody) {
        const dx = this.position.x - otherBody.position.x;
        const dy = this.position.y - otherBody.position.y;

        // Handle wrapped space
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
        return distance < (this.radius + otherBody.radius);
    }

    // Static method to create a circular masked texture for planets
    static createCircularTexture(scene, width, height) {
        // Get the original texture dimensions
        const planetTexture = scene.textures.get('planet');
        const source = planetTexture.getSourceImage();

        // Use higher resolution for better quality
        const canvas = document.createElement('canvas');
        canvas.width = 368; // Use original image dimensions
        canvas.height = 370; // Use original image dimensions
        const ctx = canvas.getContext('2d');

        // Enable image smoothing
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';

        // Clear canvas with transparency
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw a circle shape
        ctx.beginPath();
        ctx.arc(canvas.width/2, canvas.height/2, Math.min(canvas.width, canvas.height)/2, 0, Math.PI * 2);
        ctx.closePath();

        // Save and clip
        ctx.save();
        ctx.clip();

        // Draw the planet image within the clipped circle
        ctx.drawImage(source, 0, 0, canvas.width, canvas.height);

        // Restore the context state
        ctx.restore();

        // Create the texture from our canvas
        scene.textures.addCanvas('planet_circular', canvas);

        console.log('Created circular planet texture with improved quality');
    }

    // Static method to create the standard binary planet system
    static createBinarySystem() {
        const planets = [];

        // Create planet1
        const planet1 = new Planet(
            Planet.DEFAULT_ORBIT_RADIUS,
            Planet.DEFAULT_ANGULAR_VELOCITY,
            Planet.PLANET_MASS,
            Colors.RED,
            Planet.PLANET_RADIUS
        );
        planets.push(planet1);

        // Create planet2 - starts on opposite side
        const planet2 = new Planet(
            Planet.DEFAULT_ORBIT_RADIUS,
            Planet.DEFAULT_ANGULAR_VELOCITY,
            Planet.PLANET_MASS,
            Colors.RED,
            Planet.PLANET_RADIUS
        );
        planet2.angle = Math.PI;
        planet2.update(0); // Call update once to position it correctly
        planets.push(planet2);

        return planets;
    }
}