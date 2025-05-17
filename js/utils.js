console.log("utils.js starting to load");

function computeAcceleration(position, planets) {
    // Initialize acceleration vector
    const a = new Phaser.Math.Vector2(0, 0);

    // Calculate gravitational acceleration from each planet
    planets.forEach(planet => {
        // Calculate distance vector (accounting for toroidal space)
        let dx = planet.position.x - position.x;
        let dy = planet.position.y - position.y;

        // Handle toroidal wrapping for x-direction
        if (Math.abs(dx) > Constants.WIDTH / 2) {
            dx = dx > 0 ? dx - Constants.WIDTH : dx + Constants.WIDTH;
        }

        // Handle toroidal wrapping for y-direction
        if (Math.abs(dy) > Constants.HEIGHT / 2) {
            dy = dy > 0 ? dy - Constants.HEIGHT : dy + Constants.HEIGHT;
        }

        // Calculate distance vector and magnitude
        const d = new Phaser.Math.Vector2(dx, dy);
        const r = d.length();

        // Apply gravitational force if not too close
        const minDistance = planet.radius * 1.2;

        if (r > minDistance) {
            // Calculate gravitational acceleration
            const force = Constants.Gravity * planet.mass / (r * r);

            // Add to acceleration vector
            a.x += force * (d.x / r);
            a.y += force * (d.y / r);
        }
    });

    return a;
}

function checkCollision(body1, body2) {
    if (!body1 || !body2) return false;
    if (!body1.position || !body2.position) return false;

    // Check collision using shortest distance in toroidal space
    let dx = body1.position.x - body2.position.x;
    let dy = body1.position.y - body2.position.y;

    // Handle toroidal wrapping for x-direction
    if (Math.abs(dx) > Constants.WIDTH / 2) {
        if (dx > 0) {
            dx = dx - Constants.WIDTH;
        } else {
            dx = dx + Constants.WIDTH;
        }
    }

    // Handle toroidal wrapping for y-direction
    if (Math.abs(dy) > Constants.HEIGHT / 2) {
        if (dy > 0) {
            dy = dy - Constants.HEIGHT;
        } else {
            dy = dy + Constants.HEIGHT;
        }
    }

    // Calculate the distance
    const distance = Math.sqrt(dx * dx + dy * dy);

    // Check for collision
    const collisionOccurred = distance < (body1.radius + body2.radius);

    // Debug output for close encounters
    if (distance < 100 && body2 instanceof Food) {
        console.log(`Distance to food: ${distance}, Combined radii: ${body1.radius + body2.radius}, Collision: ${collisionOccurred}`);
    }

    return collisionOccurred;
}

// Function to generate food in orbit around a planet
function generateFood(count, planet) {
    const foodList = [];

    for (let i = 0; i < count; i++) {
        // Random orbit radius between 1.5x and 3x planet radius
        const orbitRadius = planet.radius * (1.5 + Math.random() * 1.5);

        // Random angle
        const angle = Math.random() * Math.PI * 2;

        // Position in orbit
        const posX = planet.position.x + Math.cos(angle) * orbitRadius;
        const posY = planet.position.y + Math.sin(angle) * orbitRadius;

        // Calculate orbital velocity for circular orbit
        // v = sqrt(G*M/r)
        const orbitSpeed = Math.sqrt(Constants.Gravity * planet.mass / orbitRadius);

        // Velocity perpendicular to radius
        const velX = -Math.sin(angle) * orbitSpeed;
        const velY = Math.cos(angle) * orbitSpeed;

        // Small random variations
        const jitter = 0.2;  // 20% variation
        const speedMultiplier = 1 + (Math.random() - 0.5) * jitter;

        // Create food
        const food = new Food(
            new Phaser.Math.Vector2(posX, posY),
            new Phaser.Math.Vector2(velX * speedMultiplier, velY * speedMultiplier),
            1 + Math.random(),  // Random mass between 1 and 2
            Colors.YELLOW
        );

        foodList.push(food);
    }

    return foodList;
}