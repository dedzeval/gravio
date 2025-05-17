class Shield {
    constructor(player) {
        this.player = player;
        this.orbitDistance = player.radius * 1.5; // Distance from player center
        this.angle = 0; // Current rotation angle

        // Use constant for arc length calculation
        this.arcLength = ((360 - Constants.SHIELD_GAP_ANGLE) * Math.PI) / 180; // In radians

        this.maxStrength = 100;
        this.currentStrength = this.maxStrength;
        this.regenerationRate = 25; // Strength points per second
        this.color = player.color || 0xFFFFFF;
        this.alpha = 0.6; // Semi-transparent
        this.thickness = 3; // Line thickness
    }

    update(dt) {
        if (!this.player || !this.player.sprite) return;

        // Get player's current shooting direction (sprite rotation)
        const shootDirection = this.player.sprite.rotation;

        // Set shield rotation to match shooting direction
        this.angle = shootDirection;

        // Update orbit distance based on current player radius
        this.orbitDistance = this.player.radius * 1.5;

        // Regenerate shield strength
        if (this.currentStrength < this.maxStrength) {
            this.currentStrength += this.regenerationRate * dt;
            this.currentStrength = Math.min(this.currentStrength, this.maxStrength); // Cap at max
        }
    }

    // In Shield class, update the draw method
    draw(graphics) {
        if (!this.player) return;

        // Base properties
        const position = this.player.position;
        const playerRadius = this.player.radius;

        let lineWidth = 0.8; // Line width for the shield (Define it earlier) - Adjusted back for visibility

        // Calculate shield properties
        const innerRadius = playerRadius * 1.3;  // Inner ring slightly away from player
        const midRadius = innerRadius + lineWidth*2; // Middle arc is one line width away
        const outerRadius = midRadius + lineWidth*2; // Outer arc is one line width away from middle

        // Calculate start and end angles for the shield arc
        // Shield has a gap defined by SHIELD_GAP_ANGLE
        const halfGapAngle = Constants.SHIELD_GAP_ANGLE * (Math.PI / 180) / 2;

        // Shield orientation follows player rotation
        const shieldDirection = this.player.sprite ? this.player.sprite.rotation : 0;

        // Gap faces direction of player rotation (front of ship)
        const gapCenterAngle = shieldDirection + Math.PI/2; // Facing direction

        const startAngle = gapCenterAngle + halfGapAngle;
        const endAngle = gapCenterAngle + Math.PI * 2 - halfGapAngle;

        // Determine number of arcs based on strength
        const strengthRatio = this.currentStrength / this.maxStrength;
        let numArcs = 0;
        if (strengthRatio > 0.9) {
            numArcs = 3;
        } else if (strengthRatio > 0.65) {
            numArcs = 2;
        } else if (strengthRatio > 0.4) {
            numArcs = 1;
        }

        // --- Use fixed transparency ---
        const transp = this.alpha; // Use the alpha defined in the constructor

        // Draw arcs based on numArcs
        if (numArcs >= 1) {
            // Draw inner arc
            graphics.lineStyle(lineWidth, this.color, transp);
            graphics.beginPath();
            graphics.arc(position.x, position.y, innerRadius, startAngle, endAngle, false);
            graphics.strokePath();
        }
        if (numArcs >= 2) {
            // Draw middle arc
            graphics.lineStyle(lineWidth, this.color, transp);
            graphics.beginPath();
            graphics.arc(position.x, position.y, midRadius, startAngle, endAngle, false);
            graphics.strokePath();
        }
        if (numArcs >= 3) {
            // Draw outer arc
            graphics.lineStyle(lineWidth, this.color, transp);
            graphics.beginPath();
            graphics.arc(position.x, position.y, outerRadius, startAngle, endAngle, false);
            graphics.strokePath();
        }

        // Optional: Add connecting lines between arcs at the ends
        // Only draw connectors if at least one arc is visible
        if (numArcs > 0) {
            const connectorRadius = (numArcs === 1) ? innerRadius : (numArcs === 2 ? midRadius : outerRadius);
            this.drawConnectors(graphics, position, innerRadius, connectorRadius, startAngle, endAngle, lineWidth, transp);
        }
    }

    // Update collision detection to use constant
    checkBeamCollision(beam) {
        if (!this.player || !beam || !beam.startPoint || !beam.endPoint) return false;

        // First check if beam intersects with shield radius at all
        const line = {
            x1: beam.startPoint.x,
            y1: beam.startPoint.y,
            x2: beam.endPoint.x,
            y2: beam.endPoint.y
        };

        // Create a temporary circle representing the shield's orbital radius
        const shieldCircle = {
            position: {
                x: this.player.position.x,
                y: this.player.position.y
            },
            radius: this.orbitDistance
        };

        // Use existing lineCircleIntersection to check basic collision
        if (!this.player.lineCircleIntersection(line, shieldCircle)) {
            return false; // No collision with shield radius
        }

        // Beam intersects with shield radius, now check if it passes through the opening
        // Calculate angle of beam intersection relative to player
        const dx = line.x2 - this.player.position.x;
        const dy = line.y2 - this.player.position.y;
        const beamAngle = Math.atan2(dy, dx);

        // Get the gap center angle and half width - use constant
        const gapAngle = (Constants.SHIELD_GAP_ANGLE * Math.PI) / 180; // Convert to radians
        const halfGap = gapAngle / 2;
        const gapCenterAngle = this.angle + Math.PI/2;

        // Calculate the angular distance between beam and gap center (normalized to [-π, π])
        let angleDiff = beamAngle - gapCenterAngle;
        while (angleDiff > Math.PI) angleDiff -= 2 * Math.PI;
        while (angleDiff < -Math.PI) angleDiff += 2 * Math.PI;

        // Check if beam is within the gap
        if (Math.abs(angleDiff) <= halfGap) {
            return false; // Beam passes through the gap
        }

        // Beam hits the shield, not the gap
        return true;
    }

    // New method to handle damage
    takeDamage(scene, position, attacker) { // CONFIRMED: Accepts scene and hit position
        if (this.currentStrength <= 0) return; // Shield already down

        const damageAmount = this.maxStrength * 0.3; // Decrease by 1/3 of max strength
        this.currentStrength -= damageAmount;
        this.currentStrength = Math.max(0, this.currentStrength); // Ensure strength doesn't go below 0

        // ADDED: Log the scene object right before calling the player's effect method
        console.log("Scene in Shield.takeDamage:", scene);
        this.player.createShieldHitEffect(scene, position);

        // *** NEW: Inform AI player it was hit ***
        if (typeof this.player.handleShieldHit === 'function') {
            this.player.handleShieldHit(attacker);
        }
    }

    // Helper method to draw connecting lines (extracted for clarity)
    drawConnectors(graphics, position, innerR, outerR, startAngle, endAngle, lineWidth, transp) {
        graphics.lineStyle(lineWidth, this.color, transp);

        // Left edge of shield
        graphics.beginPath();
        graphics.moveTo(position.x + innerR * Math.cos(startAngle), position.y + innerR * Math.sin(startAngle));
        graphics.lineTo(position.x + outerR * Math.cos(startAngle), position.y + outerR * Math.sin(startAngle));
        graphics.strokePath();

        // Right edge of shield
        graphics.beginPath();
        graphics.moveTo(position.x + innerR * Math.cos(endAngle), position.y + innerR * Math.sin(endAngle));
        graphics.lineTo(position.x + outerR * Math.cos(endAngle), position.y + outerR * Math.sin(endAngle));
        graphics.strokePath();
    }

    // Helper to check if shield is active (has strength)
    isActive() {
        return this.currentStrength > 0;
    }
}