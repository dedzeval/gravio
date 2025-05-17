class Satellite {
    static ORBIT_SPEED = 1.5; // Rotation speed in radians per second
    static SIZE_FACTOR = 1.0; // Increased from 0.3 to 3.0 (10x larger)

    constructor(player, index, totalSatellites, initialAngle = 0) {
        this.player = player;
        this.index = index;
        this.totalSatellites = totalSatellites;

        // Calculate the angle offset based on index and total count
        // This distributes satellites evenly around the orbit
        this.angleOffset = (2 * Math.PI / totalSatellites) * index;

        // Set initial angle (shared global rotation + offset)
        this.angle = initialAngle + this.angleOffset;

        // Calculate size based on player
        this.radius = player.radius * Satellite.SIZE_FACTOR;

        // Set position based on orbit distance and angle
        this.position = new Phaser.Math.Vector2(0, 0);
        this.updatePosition();

        // Set color with slight variation from player color
        this.color = this.generateColor(player.color);
    }

    generateColor(baseColor) {

        // Generate a random color with good visibility
        const colors = [
            0xFF9966,  // Lighter Orange
            0xFFCC66,  // Lighter Amber
            0x99FF99,  // Lighter Green
            0xFF99FF,  // Lighter Magenta
            0x99FFFF,  // Lighter Cyan
            0xFFFFCC   // Even Lighter Yellow
        ];

        // Pick a random color from the array
        return colors[Math.floor(Math.random() * colors.length)];
    }

    update(dt, globalRotation) {
        // Update angle using the global rotation
        this.angle = globalRotation + this.angleOffset;

        // Update position based on new angle
        this.updatePosition();
    }

    updatePosition() {
        // Use the player's pull zone factor for orbit distance
        // This makes satellites orbit exactly at the pull zone radius
        const orbitDistance = this.player.radius * Player.PULL_ZONE_FACTOR;

        // Calculate position based on orbit distance and angle
        this.position.x = this.player.position.x + Math.cos(this.angle) * orbitDistance;
        this.position.y = this.player.position.y + Math.sin(this.angle) * orbitDistance;
    }
}