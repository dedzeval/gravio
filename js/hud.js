class HUDScene extends Phaser.Scene {
    constructor() {
        super({ key: 'HUDScene', active: true });
        this.textElements = {};
        this.fragElements = [];
        this.debugMode = false;
    }

    create() {
        console.log('HUD scene created!');

        // Get reference to the main scene
        this.gameScene = this.scene.get('GameScene');

        // Setup UI text
        const textConfig = {
            fontFamily: 'Arial',
            fontSize: '10px', // Reduced from 20px
            color: '#ffffff',
            stroke: '#000000',
            strokeThickness: 2 // Reduced from 4
        };

        // Create UI elements
        this.massText = this.add.text(20, 20, 'Mass: 0', textConfig);
        this.speedText = this.add.text(20, 50, 'Speed: 0', textConfig);
        this.ejectionsText = this.add.text(20, 80, 'Ejections: 0/10', textConfig);
        this.fireText = this.add.text(20, 110, 'Fire: Ready', textConfig);

        // Add battle royale status at the top right
        const brConfig = {
            fontFamily: 'Arial',
            fontSize: '9px', // Reduced from 18px
            color: '#FF5555',
            stroke: '#000000',
            strokeThickness: 1.5 // Reduced from 3
        };

        this.playerFragText = this.add.text(
            this.cameras.main.width - 20,
            20,
            'Your Frags: 0',
            brConfig
        ).setOrigin(1, 0);

        this.playersLeftText = this.add.text(
            this.cameras.main.width - 20,
            50,
            'Players left: 0/0',
            brConfig
        ).setOrigin(1, 0);

        // Add debug mode text indicator
        this.debugModeText = this.add.text(
            this.cameras.main.width - 20,
            80,
            'Debug Mode: OFF [Press T]',
            {
                fontFamily: 'Arial',
                fontSize: '9px',
                color: '#888888',
                stroke: '#000000',
                strokeThickness: 1.5
            }
        ).setOrigin(1, 0);

        // Add debug mode toggle key
        this.debugKey = this.input.keyboard.addKey('T');
        this.input.keyboard.on('keydown-T', () => {
            this.debugMode = !this.debugMode;
            this.debugModeText.setText(`Debug Mode: ${this.debugMode ? 'ON' : 'OFF'} [Press T]`);
            this.debugModeText.setColor(this.debugMode ? '#00FF00' : '#888888');
        });

        // Create container for AI player stats
        this.textElements.aiPlayers = [];

        // Create placeholders for AI player info
        for (let i = 0; i < 10; i++) {
            const aiText = this.add.text(20, 150 + i * 25, '', textConfig);
            aiText.setScrollFactor(0);
            this.textElements.aiPlayers.push(aiText);
        }

        // Make sure this scene is positioned above the game scene
        this.scene.bringToTop();

        console.log('HUD elements created');
    }

    update() {
        // Only update if we have a reference to the game scene and it's not game over
        if (!this.gameScene || this.gameScene.gameOver) return;

        // Get player data from game scene
        const player = this.gameScene.player;
        if (!player) return;

        // Update UI text
        this.massText.setText(`Mass: ${player.mass.toFixed(2)}`);

        // Enhanced speed display with boost indicator
        const currentSpeed = player.velocity.length().toFixed(2);
        const maxSpeed = player.accelerating ? 400 : 200; // Based on Constants.RUN_MULT = 2.0
        const speedPercentage = Math.min(100, Math.floor((currentSpeed / maxSpeed) * 100));

        // Change color based on speed percentage
        let speedColor = '#FFFFFF'; // Default white
        if (speedPercentage > 90) {
            speedColor = '#FF5555'; // Red when close to max
        } else if (speedPercentage > 75) {
            speedColor = '#FFAA55'; // Orange when high
        } else if (speedPercentage > 50) {
            speedColor = '#FFFF55'; // Yellow when moderate
        }

        // Show boost indicator
        const boostIndicator = player.accelerating ? ' [BOOST]' : '';
        this.speedText.setText(`Speed: ${currentSpeed}${boostIndicator} (${speedPercentage}%)`);
        this.speedText.setColor(speedColor);

        this.ejectionsText.setText(`Ejections: ${player.ejectedMasses.length}/10`);

        // Update fire ability status
        if (player.inReloadPeriod) {
            const cooldownPercent = Math.floor((player.fireCooldown / Constants.RELOAD_COOLDOWN) * 100);
            this.fireText.setText(`Fire: RELOADING ${cooldownPercent}%`);
            this.fireText.setColor('#FF0000'); // Bright red for reload
        } else if (player.fireCooldown > 0) {
            const cooldownPercent = Math.floor((player.fireCooldown / (Constants.BEAM_DURATION + 0.5)) * 100);
            this.fireText.setText(`Fire: ${cooldownPercent}% | ${player.shotsRemaining}/${Constants.MAX_CONSECUTIVE_SHOTS} shots`);
            this.fireText.setColor('#FF5555');
        } else if (player.mass < Constants.FIRE_COST) {
            this.fireText.setText(`Fire: Need ${Constants.FIRE_COST} mass | ${player.shotsRemaining}/${Constants.MAX_CONSECUTIVE_SHOTS} shots`);
            this.fireText.setColor('#FFFF55');
        } else {
            this.fireText.setText(`Fire: Ready [SPACE] | ${player.shotsRemaining}/${Constants.MAX_CONSECUTIVE_SHOTS} shots`);
            this.fireText.setColor('#55FF55');
        }

        // Update player frag count
        this.playerFragText.setText(`Your Frags: ${player.frags || 0}`);

        // Update alive players count
        const aliveAIPlayers = this.gameScene.aiPlayers.filter(p => p.alive).length;
        const totalPlayers = aliveAIPlayers + (player.alive ? 1 : 0);
        const initialPlayers = Constants.NUM_AI_PLAYERS + 1;
        this.playersLeftText.setText(`Players Left: ${totalPlayers}/${initialPlayers}`);

        // Update AI player mass and frag info
        this.updateAIPlayerInfo();
    }

    updateAIPlayerInfo() {
        // Create a combined array of all players (AI + human player)
        const allPlayers = [...this.gameScene.aiPlayers];

        // Add main player to the list if it exists (alive or dead)
        if (this.gameScene.player) {
            // Add a special flag to identify the human player
            const humanPlayer = this.gameScene.player;
            humanPlayer.isHuman = true;
            allPlayers.push(humanPlayer);
        }

        // Sort players first by frags (highest first), then by mass (highest first)
        const sortedPlayers = allPlayers
            .slice()
            .sort((a, b) => {
                // First prioritize alive players
                if (a.alive !== b.alive) {
                    return a.alive ? -1 : 1; // Alive players go first
                }

                // Then sort by frags
                const aFrags = a.frags || 0;
                const bFrags = b.frags || 0;
                if (bFrags !== aFrags) {
                    return bFrags - aFrags; // Higher frags first
                }

                // Finally sort by mass
                return b.mass - a.mass; // Higher mass second
            })
            .slice(0, 10); // Show only top 10 players

        // Update the display for each player in the list
        for (let i = 0; i < sortedPlayers.length; i++) {
            const player = sortedPlayers[i];
            if (!player) continue;

            // Display name - "YOU" for human player, "AI #" for AI players
            const playerName = player.isHuman ? "YOU" : `AI ${this.gameScene.aiPlayers.indexOf(player) + 1}`;

            // Add death reason if available, otherwise just show [DEAD]
            let statusIndicator = '';
            if (!player.alive) {
                // Check for death reason
                if (player.deathReason) {
                    statusIndicator = ` [DEAD: ${player.deathReason}]`;
                } else {
                    // Get last player ID who hit this player, if available
                    if (player.lastHitBy) {
                        const killerName = player.lastHitBy.isHuman ? "YOU" :
                            `AI ${this.gameScene.aiPlayers.indexOf(player.lastHitBy) + 1}`;
                        statusIndicator = ` [KILLED BY ${killerName}]`;
                    } else if (player.position && (isNaN(player.position.x) || isNaN(player.position.y))) {
                        statusIndicator = ' [LOST IN SPACE]';
                    } else {
                        statusIndicator = ' [DIED]';
                    }
                }
            }

            const playerMass = player.mass.toFixed(1);
            const satelliteCount = player.satellites ? player.satellites.length : 0;
            const fragCount = player.frags || 0;

            // Calculate player speed
            const speed = isNaN(player.velocity.x) || isNaN(player.velocity.y) ?
                0 : player.velocity.length().toFixed(1);

            // Add acceleration indicator if player is boosting
            const speedIndicator = player.accelerating ?
                `<${speed}>` : // Brackets indicate boosting
                `${speed}`;   // Normal speed

            // Regular player info - now with speed
            let playerInfo = `#${i+1} ${playerName}: Mass ${playerMass} | Spd: ${speedIndicator} | Sats: ${satelliteCount} | Frags: ${fragCount}${statusIndicator}`;

            // Add debug info if test mode is enabled
            if (this.debugMode && !player.isHuman) {
                // Format coordinates and speed with fixed precision
                const posX = isNaN(player.position.x) ? "NaN" : player.position.x.toFixed(1);
                const posY = isNaN(player.position.y) ? "NaN" : player.position.y.toFixed(1);
                const speed = isNaN(player.velocity.x) || isNaN(player.velocity.y) ?
                    "NaN" : player.velocity.length().toFixed(1);

                playerInfo += ` Pos: (${posX}, ${posY}) ;Spd: ${speed}`;
            }

            // Set the text content
            this.textElements.aiPlayers[i].setText(playerInfo);

            // Rest of the code for setting colors remains the same
            if (player.alive) {
                if (player.isHuman) {
                    // Use bright white for human player
                    this.textElements.aiPlayers[i].setColor('#FFFFFF');
                    this.textElements.aiPlayers[i].setAlpha(1.0);
                } else {
                    // Use AI player's color
                    const colorHex = '#' + player.color.toString(16).padStart(6, '0');
                    this.textElements.aiPlayers[i].setColor(colorHex);
                    this.textElements.aiPlayers[i].setAlpha(1.0);
                }
            } else {
                // Use grey color for dead players
                this.textElements.aiPlayers[i].setColor('#888888');
                this.textElements.aiPlayers[i].setAlpha(0.7);
            }
        }

        // Hide unused player text elements
        for (let i = sortedPlayers.length; i < 10; i++) {
            this.textElements.aiPlayers[i].setText('');
        }

        // Remove the temporary human player flag to avoid persistence issues
        if (this.gameScene.player) {
            delete this.gameScene.player.isHuman;
        }
    }

    resize() {
        // Update text positions on resize
        if (this.playerFragText) {
            this.playerFragText.setPosition(this.cameras.main.width - 20, 20);
        }

        if (this.playersLeftText) {
            this.playersLeftText.setPosition(this.cameras.main.width - 20, 50);
        }

        if (this.debugModeText) {
            this.debugModeText.setPosition(this.cameras.main.width - 20, 80);
        }
    }
}