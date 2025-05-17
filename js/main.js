// Add this before GameScene class is defined
Colors.getRandomBrightColor = function() {
    const brightColors = [
        0xFF5555, // Bright Red
        0x55FF55, // Bright Green
        0x5555FF, // Bright Blue
        0xFFFF55, // Bright Yellow
        0xFF55FF, // Bright Magenta
        0x55FFFF, // Bright Cyan
        0xFF9955, // Bright Orange
        0xAA55FF, // Bright Purple
        0x55FFAA, // Bright Mint
        0xFFAA55  // Bright Gold
    ];

    return brightColors[Math.floor(Math.random() * brightColors.length)];
};

// Game Scene
class GameScene extends Phaser.Scene {
    constructor() {
        // Change key to allow multiple scenes
        super({ key: 'GameScene', active: true });

        // Initialize arrays to avoid undefined errors
        this.planets = [];
        this.foodList = [];
        this.foodSprites = [];
        this.satelliteSprites = [];
        this.planetSprites = [];

        // Change from single AI player to an array of AI players
        this.aiPlayers = [];
        this.aiPlayerSprites = [];

        // Add touch zoom tracking variables
        this.previousTouchDistance = 0;
        this.isZooming = false;

        this.tempGhostSprites = [];
    }

    // Keep this helper method (or ensure it exists)
    calculateGhostOffsets(position) {
        const threshold = Constants.BOUNDARY_THRESHOLD;
        const { x, y } = position;
        const offsets = [];
        let edgeX = 0; // 0 = middle, -1 = left, 1 = right
        let edgeY = 0; // 0 = middle, -1 = top, 1 = bottom

        // Determine which edges/corners are involved
        if (x < threshold) edgeX = -1;
        else if (x > Constants.WIDTH - threshold) edgeX = 1;

        if (y < threshold) edgeY = -1;
        else if (y > Constants.HEIGHT - threshold) edgeY = 1;

        // If not near any edge, return empty
        if (edgeX === 0 && edgeY === 0) return offsets;

        // Add adjacent offsets (maximum 3: horizontal, vertical, diagonal)
        // Horizontal offset
        if (edgeX !== 0) offsets.push({ x: -edgeX * Constants.WIDTH, y: 0 });
        // Vertical offset
        if (edgeY !== 0) offsets.push({ x: 0, y: -edgeY * Constants.HEIGHT });
        // Diagonal offset
        if (edgeX !== 0 && edgeY !== 0) offsets.push({ x: -edgeX * Constants.WIDTH, y: -edgeY * Constants.HEIGHT });

        return offsets;
    }

    preload() {
        // Load assets
        this.load.image('ship', 'assets/ship.png');
        this.load.image('planet', 'assets/moon1.png', { pixelArt: false });
        this.load.image('satellite', 'assets/satellite.png');

        // Add this line to load the background image
        this.load.image('background', 'assets/galaxy_bg_1.png');

        // Load star animation frames - using 12 frames as requested
        for (let i = 0; i < 12; i++) {
            const frameNum = i.toString().padStart(2, '0');
            this.load.image(`star_${frameNum}`, `assets/star_${frameNum}.png`);
        }

        console.log('Assets loading complete');
    }

    create() {
        console.log('Game scene created!');

        // Add the background as the first item so it appears behind everything
        this.background = this.add.image(Constants.WIDTH/2, Constants.HEIGHT/2, 'background');
        this.background.setDisplaySize(Constants.WIDTH, Constants.HEIGHT);
        this.background.setDepth(-100); // Ensure it's behind everything

        // Set up constants
        // starAnimationLength and starAnimationSpeed are now moved to Food class
        this.animationTime = 0;
        this.gameOver = false;

        // Set initial zoom to 4x to make the world seem larger
        this.zoomLevel = Constants.ZOOM;
        this.minZoom = 1.0;   // Increased minimum zoom
        this.maxZoom = 8.0;   // Increased maximum zoom
        this.zoomStep = 0.5;  // Larger zoom steps

        // Add flag to control pull zone circle visibility
        this.showPullZone = Constants.SHOW_PULL_ZONE;

        // Create graphics object for drawing shapes
        this.graphics = this.add.graphics();

        // Create a circular mask texture for planet - do this BEFORE initializing game objects
        Planet.createCircularTexture(this, Planet.PLANET_RADIUS*2, Planet.PLANET_RADIUS*2);

        // Initialize game objects
        this.initializeGameObjects();

        // Set up camera with 4x zoom
        this.cameras.main.startFollow(this.playerSprite);
        this.cameras.main.setZoom(this.zoomLevel);

        // Set up input
        this.cursors = this.input.keyboard.createCursorKeys();
        this.zoomKeys = {
            comma: this.input.keyboard.addKey(Phaser.Input.Keyboard.KeyCodes.COMMA),
            period: this.input.keyboard.addKey(Phaser.Input.Keyboard.KeyCodes.PERIOD),
            r: this.input.keyboard.addKey(Phaser.Input.Keyboard.KeyCodes.R),
            q: this.input.keyboard.addKey(Phaser.Input.Keyboard.KeyCodes.Q)
        };

        // Add key for toggling pull zone visibility
        this.pullZoneToggleKey = this.input.keyboard.addKey(Phaser.Input.Keyboard.KeyCodes.P);

        // Register space key for ACCELERATION (not firing)
        this.accelerateKey = this.input.keyboard.addKey(Phaser.Input.Keyboard.KeyCodes.SPACE);

        // Add mouse click handler for firing beam
        this.input.on('pointerdown', (pointer) => {
            // Only fire on left mouse button
            if (pointer.leftButtonDown() && !this.gameOver) {
                this.player.fireBeam(this);
            }
        });

        console.log('Game initialization complete');

        // Create a circular mask texture for planet
        //this.createCircularPlanetTexture();

        // Add mouse tracking for player steering
        this.input.on('pointermove', (pointer) => {
            // Convert screen coordinates to world coordinates
            this.mouseWorldX = (pointer.x / this.cameras.main.zoom) + this.cameras.main.scrollX;
            this.mouseWorldY = (pointer.y / this.cameras.main.zoom) + this.cameras.main.scrollY;

            // Debug to verify mouse tracking (less verbose)
            // console.log(`Mouse moved: screen(${pointer.x}, ${pointer.y}), world(${this.mouseWorldX}, ${this.mouseWorldY})`);
        });

        // Add multi-touch support for pinch zoom
        this.input.on('pointermove', (pointer) => {
            // Skip if we don't have at least 2 active touches
            if (this.input.pointer1.isDown && this.input.pointer2.isDown) {
                // Get the two pointers directly
                const pointer1 = this.input.pointer1;
                const pointer2 = this.input.pointer2;

                // Calculate distance between points
                const currentDistance = Phaser.Math.Distance.Between(
                    pointer1.x, pointer1.y,
                    pointer2.x, pointer2.y
                );

                // If this is the first detection of two pointers
                if (!this.isZooming) {
                    this.previousTouchDistance = currentDistance;
                    this.isZooming = true;
                    return;
                }

                // Calculate zoom change factor
                const scaleFactor = 0.01; // Adjust sensitivity
                const zoomChange = (currentDistance - this.previousTouchDistance) * scaleFactor;

                // Apply zoom change
                const newZoom = Phaser.Math.Clamp(
                    this.zoomLevel + zoomChange,
                    this.minZoom,
                    this.maxZoom
                );

                // Update camera zoom if changed
                if (newZoom !== this.zoomLevel) {
                    this.zoomLevel = newZoom;
                    this.cameras.main.setZoom(this.zoomLevel);
                }

                // Update previous distance
                this.previousTouchDistance = currentDistance;
            } else {
                // If fewer than 2 pointers are down, reset zoom state
                this.isZooming = false;
            }
        });

        // Reset zoom tracking when touches end
        this.input.on('pointerup', () => {
            if (this.input.pointersTotal < 2) {
                this.isZooming = false;
            }
        });

        // Add this to your create() method after the existing pointer handlers
        // Support for MacBook trackpad pinch gestures
        this.input.on('wheel', (pointer, gameObjects, deltaX, deltaY, deltaZ) => {
            // On MacBook trackpads, pinch gestures trigger the wheel event
            // deltaY is negative when pinching out (zoom in) and positive when pinching in (zoom out)

            // Scale factor determines zoom sensitivity
            const scaleFactor = 0.005;

            // Reverse the direction: negative delta (pinch out) should zoom in
            const zoomChange = -deltaY * scaleFactor;

            // Apply zoom change with proper boundaries
            const newZoom = Phaser.Math.Clamp(
                this.zoomLevel + zoomChange,
                this.minZoom,
                this.maxZoom
            );

            // Update camera zoom if changed
            if (newZoom !== this.zoomLevel) {
                this.zoomLevel = newZoom;
                this.cameras.main.setZoom(this.zoomLevel);
            }
        });

        // Initialize mouse position variables ONCE
        this.mouseWorldX = this.player.position.x;
        this.mouseWorldY = this.player.position.y;

        // Debug variable to track mouse issues
        this.mouseDebugTimer = 0;

        // Add this line near the end of the create method
        window.gameScene = this;

        // Create crosshair for debugging mouse coordinates
        this.crosshair = this.add.graphics();
        this.crosshair.setDepth(1000); // Ensure crosshair is drawn on top

        // Add mouse position text display for debugging
        this.mousePositionText = this.add.text(10, 10, '', {
            fontFamily: 'Arial',
            fontSize: '12px',
            color: '#FFFFFF'
        });
        this.mousePositionText.setScrollFactor(0); // Fix to camera
        this.mousePositionText.setDepth(1000);

        // Add space key for acceleration
        this.spaceKey = this.input.keyboard.addKey('SPACE');
    }

    initializeGameObjects() {
        console.log('Initializing game objects');

        // Create binary planet system using the Planet static method
        this.planets = Planet.createBinarySystem();

        // Create human player
        this.player = new Player(
            new Phaser.Math.Vector2(Constants.center_x, Constants.center_y),
            new Phaser.Math.Vector2(0, 0),
            Player.PLAYER_INITIAL_MASS,
            Colors.BLUE
        );

        const angle = Math.random() * Math.PI * 2;
        const speed = Math.random() * 20;
        this.player.velocity.x = Math.cos(angle) * speed;
        this.player.velocity.y = Math.sin(angle) * speed;

        // Add player sprite
        this.playerSprite = this.add.sprite(this.player.position.x, this.player.position.y, 'ship');
        this.playerSprite.setOrigin(0.5, 0.5);

        // Assign player sprite
        this.player.setSprite(this.playerSprite);

        // Initialize AI players array
        this.aiPlayers = [];
        this.aiPlayerSprites = [];

        // Create multiple AI players
        for (let i = 0; i < Constants.NUM_AI_PLAYERS; i++) {
            // Random position across the game area, avoiding center
            let randomX, randomY;
            let distFromCenter;

            do {
                randomX = Math.random() * Constants.WIDTH;
                randomY = Math.random() * Constants.HEIGHT;

                // Make sure AI players don't spawn too close to the center
                distFromCenter = Math.sqrt(
                    Math.pow(randomX - Constants.center_x, 2) +
                    Math.pow(randomY - Constants.center_y, 2)
                );
            } while (distFromCenter < 200); // Keep AI players away from center initially

            // Create AI player with random color
            const aiPlayer = new AIPlayer(
                new Phaser.Math.Vector2(randomX, randomY),
                new Phaser.Math.Vector2(0, 0), // Keep zero initial velocity if desired
                Player.PLAYER_INITIAL_MASS,
                Colors.getRandomBrightColor()
            );

            // Add initial random velocity (comment out these lines if you want them static)
            const aiAngle = Math.random() * Math.PI * 2;
            const aiSpeed = Math.random() * 20;
            aiPlayer.velocity.x = Math.cos(aiAngle) * aiSpeed;
            aiPlayer.velocity.y = Math.sin(aiAngle) * aiSpeed;

            // Add AI player sprite
            const aiPlayerSprite = this.add.sprite(aiPlayer.position.x, aiPlayer.position.y, 'ship');
            aiPlayerSprite.setOrigin(0.5, 0.5);
            aiPlayerSprite.setTint(aiPlayer.color);

            // Set initial rotation to match velocity direction
            aiPlayerSprite.rotation = Math.atan2(aiPlayer.velocity.y, aiPlayer.velocity.x) + Math.PI/2;

            // Add to arrays
            this.aiPlayers.push(aiPlayer);
            this.aiPlayerSprites.push(aiPlayerSprite);
        }

        // For AI players
        for (let i = 0; i < this.aiPlayers.length; i++) {
            this.aiPlayers[i].setSprite(this.aiPlayerSprites[i]);
        }

        // Initialize arrays
        this.satelliteSprites = [];
        this.planetSprites = [];
        this.foodSprites = [];

        // Add planet sprites
        for (let i = 0; i < this.planets.length; i++) {
            const planet = this.planets[i];
            // Use planet's createSprite method
            const sprite = planet.createSprite(this);

            // Store reference to sprite
            this.planetSprites.push({ planet, sprite });
        }

        // Generate food - now with random distribution across the game area
        try {
            // Create a mix of food: some around planets, some randomly distributed
            this.foodList = [
                // Some food still around planets (50 each)
                ...Food.generateAroundPlanet(Constants.FOOD_COUNT, this.planets[0]),
                ...Food.generateAroundPlanet(Constants.FOOD_COUNT, this.planets[1]),
                // Add 200 randomly distributed food across the game area
                ...Food.generateRandom(Constants.FOOD_COUNT*2)
            ];

            console.log(`Created ${this.foodList.length} food items`);

            // Assign random animation phases
            this.foodList.forEach(food => {
                food.animationPhase = Math.random();
            });
        } catch (e) {
            console.error("Error generating food:", e);
            this.foodList = [];
        }
    }

    update(time, delta) {
        // Convert delta to seconds with a time scaling factor to slow down the game
        const TIME_SCALE = 0.25; // Slow down the physics by 4x
        const dt = delta / 1000.0 * TIME_SCALE;
        this.animationTime += delta / 1000.0; // Keep animation time normal speed

        if (this.gameOver) {
            // Handle game over input
            if (Phaser.Input.Keyboard.JustDown(this.zoomKeys.r)) {
                this.scene.restart();
            } else if (Phaser.Input.Keyboard.JustDown(this.zoomKeys.q)) {
                this.game.destroy(true);
                return;
            }
            return;
        }

        // Handle zoom controls
        if (Phaser.Input.Keyboard.JustDown(this.zoomKeys.comma)) {
            this.zoomLevel = Math.max(this.minZoom, this.zoomLevel - this.zoomStep);
            this.cameras.main.setZoom(this.zoomLevel);
        } else if (Phaser.Input.Keyboard.JustDown(this.zoomKeys.period)) {
            this.zoomLevel = Math.min(this.maxZoom, this.zoomLevel + this.zoomStep);
            this.cameras.main.setZoom(this.zoomLevel);
        }

        // Add game over on Q keypress
        if (Phaser.Input.Keyboard.JustDown(this.zoomKeys.q)) {
            this.displayGameOver();
            return;
        }

        // Toggle pull zone visibility when P is pressed
        if (Phaser.Input.Keyboard.JustDown(this.pullZoneToggleKey)) {
            this.showPullZone = !this.showPullZone;
            console.log(`Pull zone visibility: ${this.showPullZone ? 'ON' : 'OFF'}`);
        }

        // REMOVE THIS SECTION - spacebar no longer fires beam
        /*
        // Handle fire key
        if (Phaser.Input.Keyboard.JustDown(this.fireKey)) {
            this.player.fireBeam(this);
        }
        */

        // IMPORTANT: Only calculate mouse position ONCE per frame
        const pointer = this.input.activePointer;

        // FIX: Use Phaser's built-in getWorldPoint method instead of manual calculation
        // This properly accounts for zoom, scroll, rotation and other camera transformations
        const worldPoint = this.cameras.main.getWorldPoint(pointer.x, pointer.y);
        this.mouseWorldX = worldPoint.x;
        this.mouseWorldY = worldPoint.y;

        // Debug mouse coordinates if needed
        if (this.mouseDebugTimer === 1) {
            console.log("----MOUSE DEBUG----");
            console.log(`Screen pos: (${pointer.x.toFixed(1)}, ${pointer.y.toFixed(1)})`);
            console.log(`Camera: zoom=${this.cameras.main.zoom}, scroll=(${this.cameras.main.scrollX.toFixed(1)}, ${this.cameras.main.scrollY.toFixed(1)})`);
            console.log(`World pos: (${this.mouseWorldX.toFixed(1)}, ${this.mouseWorldY.toFixed(1)})`);
            console.log(`Player pos: (${this.player.position.x.toFixed(1)}, ${this.player.position.y.toFixed(1)})`);
        }

        // Debug timer for mouse positioning
        if (this.mouseDebugTimer === undefined) {
            this.mouseDebugTimer = 0;
        }
        this.mouseDebugTimer--;
        if (this.mouseDebugTimer < 0) {
            this.mouseDebugTimer = 120; // Output debug info every ~2 seconds
        }

        if (!this.gameOver) {
            // Calculate angle to mouse cursor ONCE
            const dx = this.mouseWorldX - this.player.position.x;
            const dy = this.mouseWorldY - this.player.position.y;

            // Debug rotation values
            if (this.mouseDebugTimer === 1) {
                console.log(`Delta: (${dx.toFixed(2)}, ${dy.toFixed(2)})`);
                console.log(`Angle to mouse: ${Math.atan2(dy, dx).toFixed(2)} rad`);
                console.log(`Current rotation: ${this.playerSprite.rotation.toFixed(2)} rad`);
            }

            // Only update rotation if mouse is not exactly on player (prevents NaN)
            if (Math.abs(dx) > 0.001 || Math.abs(dy) > 0.001) {
                // Calculate target angle
                const targetAngle = Math.atan2(dy, dx);

                // FIXED: Add PI (180 degrees) to flip the player sprite
                // Now using targetAngle + Math.PI + Math.PI/2 which equals targetAngle + 3*Math.PI/2
                this.playerSprite.rotation = targetAngle + 3*Math.PI/2;

                // Debug the new rotation
                if (this.mouseDebugTimer === 1) {
                    console.log(`New rotation set: ${this.playerSprite.rotation.toFixed(2)} rad`);
                }
            }

            // Player Input - ENHANCED MOVEMENT SYSTEM
            // Calculate current velocity direction for reference
            const currentVelocity = this.player.velocity.clone();
            const currentSpeed = currentVelocity.length();
            let movementDirection;

            // Get movement direction from current velocity or sprite rotation
            if (currentSpeed > 0.1) {
                movementDirection = currentVelocity.normalize();
            } else {
                // If barely moving, use the direction the sprite is facing (now mouse-controlled)
                const playerAngle = this.playerSprite.rotation - Math.PI/2;
                movementDirection = new Phaser.Math.Vector2(Math.cos(playerAngle), Math.sin(playerAngle));
            }

            // Calculate perpendicular vectors for side movement
            const perpRight = new Phaser.Math.Vector2(-movementDirection.y, movementDirection.x);
            const perpLeft = new Phaser.Math.Vector2(movementDirection.y, -movementDirection.x);

            // Left/right keys now only control strafing, not rotation
            if (this.cursors.left.isDown) {
                // SIDE ACCELERATION only (no rotation)
                if (this.player.mass > Player.MIN_MASS * 1.5) {
                    const strafeStrength = 0.7; // Side thrust is 70% of forward thrust
                    const strafeDirection = perpRight;
                    const ejectedFood = this.player.ejectMass(strafeDirection, dt * strafeStrength, this.planets);
                    if (ejectedFood) this.foodList.push(ejectedFood);
                }
            }

            if (this.cursors.right.isDown) {
                // SIDE ACCELERATION only (no rotation)
                if (this.player.mass > Player.MIN_MASS * 1.5) {
                    const strafeStrength = 0.7; // Side thrust is 70% of forward thrust
                    const strafeDirection = perpLeft;
                    const ejectedFood = this.player.ejectMass(strafeDirection, dt * strafeStrength, this.planets);
                    if (ejectedFood) this.foodList.push(ejectedFood);
                }
            }

            // FORWARD THRUST: Calculate forward direction based on ship rotation (which is now mouse-controlled)
            const playerAngle = this.playerSprite.rotation - Math.PI/2;
            const forwardX = Math.cos(playerAngle);
            const forwardY = Math.sin(playerAngle);

            // Rest of movement code is unchanged
            // Add spacebar as an alternative to up arrow for thrust
            if (this.cursors.up.isDown || this.input.keyboard.checkDown(this.accelerateKey)) {
                // THRUST: Eject mass BACKWARD to move FORWARD
                const thrustDirection = new Phaser.Math.Vector2(-forwardX, -forwardY);
                const ejectedFood = this.player.ejectMass(thrustDirection, dt, this.planets);
                if (ejectedFood) this.foodList.push(ejectedFood);
            }
        }

        // Update game objects
        this.planets.forEach(planet => planet.update(dt));

        // Update food with player pull influence
        this.foodList.forEach(food => food.update(dt, this.planets, this, this.animationTime));

        // Update human player - updateSatellites now happens inside player.update()
        this.player.update(dt, this.planets);

        // Update bots
        for (let i = 0; i < this.aiPlayers.length; i++) {
            const aiPlayer = this.aiPlayers[i];
            if (!aiPlayer.alive) continue;
            aiPlayer.update(dt, this.planets);
        }

        /*
        // Check for NaN coordinate checks (run this every few frames)
        if (this.frameCount % 30 === 0) {  // Check every 30 frames
            // Check human player
            if (this.player && this.player.alive &&
                (isNaN(this.player.position.x) || isNaN(this.player.position.y))) {
                this.player.die("LOST IN SPACE", null);
                this.displayGameOver("LOST IN SPACE");
            }

            // Check bots
            for (const aiPlayer of this.aiPlayers) {
                if (aiPlayer.alive &&
                    (isNaN(aiPlayer.position.x) || isNaN(aiPlayer.position.y))) {
                    aiPlayer.die("LOST IN SPACE", null);
                }
            }
        }
        */

        // Collision Detection for food consumption (for both players)
        this.player.collectFood(this.foodList, this.foodSprites, this);

        // Bots collect food
        for (const aiPlayer of this.aiPlayers) {
            aiPlayer.collectFood(this.foodList, this.foodSprites, this);
        }

        // Check shield collisions between all players
        this.checkAllShieldCollisions();

        // --- Boundary Bouncing ---
        if (this.player.alive) {
            this.handleBoundaryBounce(this.player);
        }
        this.aiPlayers.forEach(ai => {
            if (ai.alive) this.handleBoundaryBounce(ai);
        });

        // Update visuals
        this.updateVisuals();

        // Draw a simple crosshair with two thin crossed lines
        this.crosshair.clear();
        this.crosshair.lineStyle(1/this.cameras.main.zoom, 0xFF0000, 1); // Thinner line (1px adjusted for zoom)

        // Draw crosshair lines that stay the same size regardless of zoom
        const crosshairSize = 15 / this.cameras.main.zoom; // Make crosshair size consistent at all zoom levels

        // Draw horizontal line
        this.crosshair.beginPath();
        this.crosshair.moveTo(this.mouseWorldX - crosshairSize, this.mouseWorldY);
        this.crosshair.lineTo(this.mouseWorldX + crosshairSize, this.mouseWorldY);

        // Draw vertical line
        this.crosshair.moveTo(this.mouseWorldX, this.mouseWorldY - crosshairSize);
        this.crosshair.lineTo(this.mouseWorldX, this.mouseWorldY + crosshairSize);
        this.crosshair.strokePath();

        // Update position text display
        this.mousePositionText.setText(
            `Screen: (${pointer.x.toFixed(0)},${pointer.y.toFixed(0)}) | ` +
            `World: (${this.mouseWorldX.toFixed(0)},${this.mouseWorldY.toFixed(0)}) | ` +
            `Player: (${this.player.position.x.toFixed(0)},${this.player.position.y.toFixed(0)})`
        );

        // Check space key for player acceleration
        if (this.player && this.player.alive) {
            this.player.accelerating = this.spaceKey.isDown;
        }
    }

    updateVisuals() {
        // --- Cleanup ---
        this.graphics.clear();

        // Only draw borders if enabled in constants
        if (Constants.BORDER_SHOWN) {
            // Draw game boundary
            this.graphics.lineStyle(2, 0xFFFFFF, 0.8);
            this.graphics.strokeRect(0, 0, Constants.WIDTH, Constants.HEIGHT);
        }

        // --- Render Entities ---
        // Update sprite positions and draw graphics

        // Player
        if (this.player.alive && this.player.sprite) {
            this.player.sprite.setPosition(this.player.position.x, this.player.position.y);
            this.player.sprite.visible = true; // Ensure visible
            this.player.drawBeam(this.graphics);
            this.player.drawPullZone(this.graphics, this.animationTime);
            this.player.drawShield(this.graphics);
        } else if (this.player.sprite) {
            this.player.sprite.visible = false; // Hide if dead
        }

        // Bots
        for (let i = 0; i < this.aiPlayers.length; i++) {
            const bot = this.aiPlayers[i];
            const botSprite = this.aiPlayerSprites[i];
            if (bot.alive && botSprite) {
                botSprite.setPosition(bot.position.x, bot.position.y);
                botSprite.setDisplaySize(bot.radius * 2, bot.radius * 2); // Update size based on mass
                botSprite.visible = true; // Ensure visible
                bot.drawBeam(this.graphics);
                bot.drawShield(this.graphics);
            } else if (botSprite) {
                botSprite.visible = false; // Hide if dead
            }
        }

        // Planets
        this.planetSprites.forEach(ps => {
            ps.sprite.setPosition(ps.planet.position.x, ps.planet.position.y);
        });

        // Food (handled by Food.update) - ensure sprites are positioned if needed
        // (Assuming Food.update handles sprite visibility/position)
        this.foodList.forEach(food => {
            if (food.sprite) {
                 food.sprite.setPosition(food.position.x, food.position.y);
                 // Visibility is likely handled in Food.update based on collected status
            }
        });

        // --- Draw Crosshair and Debug Text ---
        this.crosshair.clear();
        this.crosshair.lineStyle(1 / this.cameras.main.zoom, 0xFF0000, 1);
        const crosshairSize = 15 / this.cameras.main.zoom;
        this.crosshair.beginPath();
        this.crosshair.moveTo(this.mouseWorldX - crosshairSize, this.mouseWorldY);
        this.crosshair.lineTo(this.mouseWorldX + crosshairSize, this.mouseWorldY);
        this.crosshair.moveTo(this.mouseWorldX, this.mouseWorldY - crosshairSize);
        this.crosshair.lineTo(this.mouseWorldX, this.mouseWorldY + crosshairSize);
        this.crosshair.strokePath();

        this.mousePositionText.setText(
            `Screen: (${this.mouseWorldX.toFixed(0)},${this.mouseWorldY.toFixed(0)}) | ` +
            `Player: (${this.player.position.x.toFixed(0)},${this.player.position.y.toFixed(0)})`
        );
    }

    // New method for handling boundary bouncing
    handleBoundaryBounce(entity) {
        const radius = entity.radius || 10; // Use entity radius or a default
        const bounceFactor = 0.5; // Energy loss on bounce (0 = no bounce, 1 = perfect bounce)

        // Check horizontal boundaries
        if (entity.position.x - radius < 0) {
            entity.position.x = radius; // Clamp position
            entity.velocity.x = Math.abs(entity.velocity.x) * bounceFactor; // Reverse and dampen velocity
        } else if (entity.position.x + radius > Constants.WIDTH) {
            entity.position.x = Constants.WIDTH - radius; // Clamp position
            entity.velocity.x = -Math.abs(entity.velocity.x) * bounceFactor; // Reverse and dampen velocity
        }

        // Check vertical boundaries
        if (entity.position.y - radius < 0) {
            entity.position.y = radius; // Clamp position
            entity.velocity.y = Math.abs(entity.velocity.y) * bounceFactor; // Reverse and dampen velocity
        } else if (entity.position.y + radius > Constants.HEIGHT) {
            entity.position.y = Constants.HEIGHT - radius; // Clamp position
            entity.velocity.y = -Math.abs(entity.velocity.y) * bounceFactor; // Reverse and dampen velocity
        }
    }

    checkAllShieldCollisions() {
        // Check collision between human player and all AI players
        if (this.player && this.player.alive) {
            for (const aiPlayer of this.aiPlayers) {
                if (aiPlayer.alive && this.player.beamTimer > 0) {
                    // Check collision only if shield is active
                    if (aiPlayer.shield.isActive() && aiPlayer.shield.checkBeamCollision(this.player.beam)) {
                        // Beam hits shield
                        // Pass scene (this) and beam endpoint position
                        const hitPosition = this.player.beam ? this.player.beam.endPoint : aiPlayer.position; // Use beam end or fallback
                        console.log("Scene context in checkAllShieldCollisions (Player vs AI):", this); // Keep log for verification
                        aiPlayer.shield.takeDamage(this, hitPosition);
                        this.player.beamTimer = 0; // Stop beam on hit
                        const strengthRatio = aiPlayer.shield.currentStrength / aiPlayer.shield.maxStrength;
                        if (strengthRatio<0.1){
                            aiPlayer.die("SHOT BY PLAYER", this.player);
                        }
                    } else if (this.player.checkBeamHit(aiPlayer)) {
                        // Beam hits player body (passed through shield gap or shield down)
                        aiPlayer.die("SHOT BY PLAYER", this.player);
                    }
                }
            }
        }

        // Check collisions between all bots
        for (let i = 0; i < this.aiPlayers.length; i++) {
            const ai1 = this.aiPlayers[i];
            if (!ai1 || !ai1.alive || ai1.beamTimer <= 0) continue;

            for (let j = i + 1; j < this.aiPlayers.length; j++) {
                const ai2 = this.aiPlayers[j]; // The player being shot at
                if (!ai2 || !ai2.alive) continue;

                // Check if ai1's beam hits ai2's shield
                if (ai2.shield.isActive() && ai2.shield.checkBeamCollision(ai1.beam)) {
                    // Pass scene (this) and beam endpoint position
                    const hitPosition = ai1.beam ? ai1.beam.endPoint : ai2.position; // Use beam end or fallback
                    console.log("Scene context in checkAllShieldCollisions (AI vs AI):", this); // Keep log for verification
                    ai2.shield.takeDamage(this, hitPosition);
                    ai1.beamTimer = 0; // Stop beam on hit
                } else if (ai1.checkBeamHit(ai2)) {
                    // Beam hits player body
                    ai2.die("SHOT BY BOT", ai1);
                }
            }
        }
    }

    displayGameOver(message = 'GAME OVER') {
        this.gameOver = true;

        // Display game over message
        const centerX = Constants.WIDTH / 2;
        const centerY = Constants.HEIGHT / 2;

        const gameOverText = this.add.text(centerX, centerY - 50, message, {
            fontFamily: 'Arial',
            fontSize: '64px',
            color: '#ff0000',
            align: 'center'
        });
        gameOverText.setOrigin(0.5);

        const scoreText = this.add.text(centerX, centerY + 50, `Final Score: ${Math.floor(this.player.mass)}`, {
            fontFamily: 'Arial',
            fontSize: '32px',
            color: '#ffffff',
            align: 'center'
        });
        scoreText.setOrigin(0.5);

        const restartText = this.add.text(centerX, centerY + 120, 'Click to restart', {
            fontFamily: 'Arial',
            fontSize: '24px',
            color: '#ffff00',
            align: 'center'
        });
        restartText.setOrigin(0.5);

        // Add click handler to restart
        this.input.once('pointerup', () => {
            this.scene.restart();
        });
    }

    resetGame() {
        this.gameOver = false;
        // Game will restart through scene.restart()

        // Clean up satellite sprites
        if (this.satelliteSprites) {
            this.satelliteSprites.forEach(sprite => sprite.destroy());
            this.satelliteSprites = [];
        }
    }

    resize() {
        console.log('Window resized, updating UI positions');
    }

    // Update respawnAIPlayer (now respawnBot) to handle battle royale mode
    respawnAIPlayer(index, killerPlayer = null) {
        // In battle royale mode, don't respawn - mark as dead instead
        if (index >= 0 && index < this.aiPlayers.length) {
            const deadPlayer = this.aiPlayers[index];

            // Call die method to clean up satellites and mark as dead
            deadPlayer.die("respawned", killerPlayer);

            // If we know who killed this player, increment their frag count
            if (killerPlayer) {

                // Create a text popup at the death location
                this.createFragPopup(deadPlayer.position.x, deadPlayer.position.y, killerPlayer);
            }

            // Hide the sprite
            if (this.aiPlayerSprites && this.aiPlayerSprites[index]) {
                this.aiPlayerSprites[index].visible = false;
            }

            // Check if game should end (player is last one standing)
            this.checkBattleRoyaleStatus();
        }
    }

    // Add method to display frag popup
    createFragPopup(x, y, killerPlayer) {
        const popupText = this.add.text(x, y, '+1 FRAG', {
            fontFamily: 'Arial',
            fontSize: '24px',
            color: '#FFFF00',
            stroke: '#000000',
            strokeThickness: 4
        }).setOrigin(0.5);

        // Animate the popup
        this.tweens.add({
            targets: popupText,
            y: y - 50,
            alpha: 0,
            duration: 1500,
            ease: 'Power1',
            onComplete: () => {
                popupText.destroy();
            }
        });
    }

    // Add method to check if battle royale should end
    checkBattleRoyaleStatus() {
        // Count alive bots
        const aliveAICount = this.aiPlayers.filter(ai => ai.alive).length;

        // If player is the only one left, they win
        if (aliveAICount === 0 && this.player.alive) {
            this.displayGameOver("VICTORY ROYALE! You're the last one standing!");
        }

        // Update remaining player count in HUD if it exists
        if (this.scene.get('HUDScene')) {
            const hudScene = this.scene.get('HUDScene');
            if (hudScene.updatePlayerCount) {
                hudScene.updatePlayerCount(aliveAICount + (this.player.alive ? 1 : 0));
            }
        }
    }
}
// Configuration for Phaser game
const config = {
    type: Phaser.AUTO,
    width: Constants.WIDTH,
    height: Constants.HEIGHT,
    backgroundColor: '#000000',
    scene: [GameScene, HUDScene], // Add both scenes
    physics: {
        default: 'arcade',
        arcade: {
            debug: false
        }
    },
    scale: {
        mode: Phaser.Scale.RESIZE,
        autoCenter: Phaser.Scale.CENTER_BOTH
    }
};

// Initialize Phaser
const game = new Phaser.Game(config);