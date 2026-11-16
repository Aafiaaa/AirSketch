
import time
import math

from particles.particle_engine import ParticleEngine


class ParticleEffects:

    def __init__(self, width=1280, height=720):

        self.engine = ParticleEngine(width, height)

        self.previous_time = time.monotonic()

        self.last_position = None

    @property
    def style(self):

        return self.engine.style

    def change_style(self):

        return self.engine.change_style()

    def draw_trail(self, point, color, brush_size):

        x, y = point

        # First point in a new trail
        if self.last_position is None:

            self.engine.emit_trail(
                x,
                y,
                color,
                brush_size
            )

            self.last_position = point
            return

        previous_x, previous_y = self.last_position

        dx = x - previous_x
        dy = y - previous_y

        distance = math.hypot(dx, dy)

        # Avoid connecting two distant, separate strokes
        if distance > 180:

            self.last_position = point

            self.engine.emit_trail(
                x,
                y,
                color,
                brush_size
            )

            return

        # Interpolate points along the fingertip movement.
        # Smaller spacing creates a denser, continuous trail.
        spacing = 3

        steps = max(
            1,
            math.ceil(distance / spacing)
        )

        for i in range(1, steps + 1):

            ratio = i / steps

            trail_x = int(previous_x + dx * ratio)
            trail_y = int(previous_y + dy * ratio)

            self.engine.emit_trail(
                trail_x,
                trail_y,
                color,
                brush_size
            )

        self.last_position = point

    def color_burst(self, point, color):

        if point is None:
            return

        self.engine.emit_burst(
            point[0],
            point[1],
            color,
            count=90
        )

    def update_and_render(self, frame):

        current_time = time.monotonic()

        dt = current_time - self.previous_time

        self.previous_time = current_time

        self.engine.update(dt)

        return self.engine.render(frame)

    def clear(self):

        self.engine.clear()
        self.last_position = None
