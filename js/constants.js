const Constants = {
    WIDTH: 1200,
    HEIGHT: 800,
    Gravity: 200,
    center_x: 600,
    center_y: 400,
    SHOW_PULL_ZONE: false,
    ZOOM: 2.5,
    SCALE_FACTOR: 4, // Scale-down factor to make the world appear larger
    BORDER_SHOWN: false,      // Toggle to show/hide game scene border

    NUM_AI_PLAYERS: 20, // Number of AI players in the swarm
    FIRE_COST: 1, // Mass required to fire beam
    BEAM_LENGTH_FACTOR: 20, // Beam length relative to player size
    BEAM_DURATION: 0.1, // Beam duration in seconds

    // Fire cooldown constants
    MAX_CONSECUTIVE_SHOTS: 5,
    RELOAD_COOLDOWN: 1.0,
    SHOT_COOLDOWN_MULTIPLIER: 0.7,  // Initial multiplier value
    MULTIPLIER_COOLDOWN: 1.0,        // Time in seconds before multiplier resets

    RUN_MULT: 2.0, // Speed multiplier when running (space pressed)
    FOOD_COUNT: 100,

    // Toggle for showing seamless world boundaries
    SEAMLESS_BOUNDARIES: true,     // Enable by default
    BOUNDARY_THRESHOLD: 100,       // How close to edge objects need to be to show duplicates

    // Satellite properties
    SATELLITE_STEP: 20,          // One satellite per 20 mass
    SATELLITE_BLOCK_CHANCE: 1.0, // 100% chance to block beam
    SATELLITE_DESTRUCTION: true, // Satellite is destroyed when hit by beam
    SATELLITE_HITBOX: 1.2,       // Multiplier for satellite hitbox size

    // Bot engagement threshold as multiple of bot radius
    BOT_ENGAGEMENT_RADIUS_MULT: 10,

    // Fixed speed values independent of mass
    BASE_SPEED: 200,  // Normal movement speed
    RUN_SPEED: 500,   // Speed when accelerating (space pressed)

    // Shield properties
    SHIELD_GAP_ANGLE: 90, // Shield opening angle in degrees

    // Bot rotation speed
    BOT_MAX_ROTATION_SPEED: 30 * (Math.PI / 180) // 30 degrees per second in radians
};

const Colors = {
    BLACK: 0x000000,
    BLUE: 0x0000FF,
    RED: 0xFF0000,
    GREEN: 0x00FF00,
    WHITE: 0xFFFFFF,
    YELLOW: 0xFFFF00,
    PURPLE: 0x800080,
    CYAN: 0x00FFFF
};