
import cv2
import math
import random
import numpy as np

from particles.particle import Particle


class ParticleEngine:

    def __init__(self, width=1280, height=720):

        self.width = width
        self.height = height

        self.particles = []

        # Increased particle capacity
        self.max_particles = 10000

        self.styles = [
            "GLOW",
            "FIRE",
            "MAGIC",
            "RAINBOW"
        ]

        self.style_index = 0
        self.style = self.styles[self.style_index]

        self.rainbow_index = 0

    def change_style(self):

        self.style_index = (
            self.style_index + 1
        ) % len(self.styles)

        self.style = self.styles[self.style_index]

        return self.style

    def get_color(self, base_color):

        if self.style == "GLOW":

            # Mostly the selected colour, with occasional
            # bright white-blue sparkles.
            return random.choice([
                base_color,
                base_color,
                base_color,
                (255, 255, 255),
                (255, 240, 210)
            ])

        elif self.style == "FIRE":

            return random.choice([
                (0, 45, 255),
                (0, 90, 255),
                (0, 150, 255),
                (0, 210, 255),
                (40, 230, 255)
            ])

        elif self.style == "MAGIC":

            return random.choice([
                (255, 80, 255),
                (255, 120, 180),
                (220, 80, 255),
                (255, 200, 120),
                (255, 255, 255)
            ])

        elif self.style == "RAINBOW":

            colors = [
                (0, 0, 255),
                (0, 127, 255),
                (0, 255, 255),
                (0, 255, 0),
                (255, 0, 0),
                (255, 0, 127),
                (255, 0, 255)
            ]

            color = colors[
                self.rainbow_index % len(colors)
            ]

            self.rainbow_index += 1

            return color

        return base_color

    
    def emit_trail(self, x, y, base_color, brush_size=10):

        available = self.max_particles - len(self.particles)

        if available <= 0:
            return

        # Increase particle density
        particle_count = min(
            8,
            6 + brush_size // 10,
            available
        )

        # Wider particle distribution
        spread = max(14, brush_size * 1.5)

        for _ in range(particle_count):

            # Spread particles around the fingertip
            offset_x = random.gauss(0, spread * 0.55)
            offset_y = random.gauss(0, spread * 0.55)

            particle_x = x + offset_x
            particle_y = y + offset_y

            angle = random.uniform(0, math.tau)
            speed = random.uniform(3, 28)

            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed

            color = self.get_color(base_color)

            size = random.uniform(0.6, 1.4)

            if random.random() < 0.08:
                size = random.uniform(1.4, 1.8)
                color = (255, 255, 255)

            lifetime = random.uniform(1.5, 2.5)

            self.particles.append(
                Particle(
                    particle_x,
                    particle_y,
                    vx,
                    vy,
                    color,
                    size,
                    lifetime
                )
            )


    def emit_burst(self, x, y, base_color, count=90):

        available = self.max_particles - len(self.particles)
        count = min(count, available)

        for _ in range(count):

            angle = random.uniform(0, math.tau)
            speed = random.uniform(30, 150)

            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed

            self.particles.append(
                Particle(
                    x,
                    y,
                    vx,
                    vy,
                    self.get_color(base_color),
                    random.uniform(0.7, 1.7),
                    random.uniform(1.3, 2.2)
                )
            )

    def update(self, dt):

        dt = min(max(dt, 0), 0.05)

        for particle in self.particles:
            particle.update(dt)

        self.particles = [
            particle
            for particle in self.particles
            if particle.alive
        ]

    def render(self, frame):

        # IMPORTANT:
        # This is a black, transparent-style overlay.
        # It does not contain a copy of the webcam frame.
        overlay = np.zeros_like(frame)

        for particle in self.particles:

            x, y = particle.get_position()

            if not (
                0 <= x < self.width
                and 0 <= y < self.height
            ):
                continue

            alpha = particle.get_alpha()

            if alpha <= 0:
                continue

            color = tuple(
                int(channel * alpha)
                for channel in particle.color
            )

            # Most particles are single-pixel points.
            radius = max(1, round(particle.size))

            cv2.circle(
                overlay,
                (x, y),
                radius,
                color,
                -1,
                cv2.LINE_AA
            )

        # Create glow using ONLY the particle overlay.
        glow = cv2.GaussianBlur(
            overlay,
            (0, 0),
            sigmaX=2.5,
            sigmaY=2.5
        )

        # Add a subtle transparent black tint.
        black_overlay = np.zeros_like(frame)

        display = cv2.addWeighted(
            frame,
            0.58,
            black_overlay,
            0.42,
            0
        )

        # Add particle glow.
        display = cv2.addWeighted(
            display,
            1.0,
            glow,
            0.75,
            0
        )

        # Add sharp particle centres on top.
        display = cv2.addWeighted(
            display,
            1.0,
            overlay,
            0.85,
            0
        )

        return display

    def clear(self):

        self.particles.clear()
