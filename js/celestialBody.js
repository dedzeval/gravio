class CelestialBody {
    constructor(position, velocity, mass, color, radius = null) {
        this.position = position;
        this.velocity = velocity;
        this.mass = mass;
        this.color = color;
        this.K = 2; // Radius scaling factor

        // Calculate radius based on mass if not provided
        if (radius === null) {
            this.radius = Math.max(this.K * Math.sqrt(this.mass), 1);
        } else {
            this.radius = radius;
        }
    }

    update(dt, planets) {
        // Apply gravitational forces
        this.applyGravity(dt, planets);

        // Update position based on velocity
        this.position.x += this.velocity.x * dt;
        this.position.y += this.velocity.y * dt;

        // Handle toroidal wrapping
        this.position.x = (this.position.x + Constants.WIDTH) % Constants.WIDTH;
        this.position.y = (this.position.y + Constants.HEIGHT) % Constants.HEIGHT;

        // Ensure position wrapping works correctly
        this.handleWrapping();
    }

    applyGravity(dt, planets) {
        // For each planet, calculate gravitational force
        planets.forEach(planet => {
            // Skip self-attraction
            if (planet === this) return;

            // Calculate vector to planet
            const dx = planet.position.x - this.position.x;
            const dy = planet.position.y - this.position.y;

            // Handle toroidal wrapping - find the shortest distance
            let wrappedDx = dx;
            if (Math.abs(dx) > Constants.WIDTH / 2) {
                wrappedDx = dx - Math.sign(dx) * Constants.WIDTH;
            }

            let wrappedDy = dy;
            if (Math.abs(dy) > Constants.HEIGHT / 2) {
                wrappedDy = dy - Math.sign(dy) * Constants.HEIGHT;
            }

            // Calculate distance squared
            const distanceSquared = wrappedDx * wrappedDx + wrappedDy * wrappedDy;
            // Add small value to prevent division by zero and excessive force at close distances
            const distance = Math.sqrt(distanceSquared) + 10; // Increased minimum distance to reduce gravity at close range

            // Calculate gravitational force magnitude (G*m1*m2/r^2)
            const forceMagnitude = Constants.Gravity * this.mass * planet.mass / distanceSquared;

            // Apply force limiter to prevent extreme acceleration
            const maxForce = 5000; // Set maximum force
            const limitedForce = Math.min(forceMagnitude, maxForce);

            // Calculate force components
            const forceX = limitedForce * wrappedDx / distance;
            const forceY = limitedForce * wrappedDy / distance;

            // Apply force (F = ma -> a = F/m)
            this.velocity.x += forceX / this.mass * dt;
            this.velocity.y += forceY / this.mass * dt;
        });
    }
}