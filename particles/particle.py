
class Particle:

    def __init__(self, x, y, vx, vy, color, size, lifetime):
        self.x = float(x)
        self.y = float(y)

        self.vx = float(vx)
        self.vy = float(vy)

        self.color = color
        self.size = float(size)

        self.lifetime = float(lifetime)
        self.max_lifetime = float(lifetime)

        self.alive = True

    def update(self, dt):

        self.x += self.vx * dt
        self.y += self.vy * dt

        # Gentle movement instead of strong falling gravity
        self.vx *= 0.995
        self.vy *= 0.995

        self.lifetime -= dt

        if self.lifetime <= 0:
            self.alive = False

    def get_alpha(self):

        if self.max_lifetime <= 0:
            return 0.0

        remaining = self.lifetime / self.max_lifetime

        # Smooth fade-out
        return max(0.0, min(1.0, remaining))

    def get_position(self):

        return int(self.x), int(self.y)
