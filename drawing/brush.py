
class Brush:

    def __init__(self, size=10, smoothing=0.35):

        self.size = size
        self.smoothing = smoothing

        self.previous_position = None
        self.current_position = None

    def smooth_position(self, new_position):

        if self.current_position is None:

            self.current_position = new_position

        else:

            x = int(
                self.smoothing * new_position[0]
                + (1 - self.smoothing) * self.current_position[0]
            )

            y = int(
                self.smoothing * new_position[1]
                + (1 - self.smoothing) * self.current_position[1]
            )

            self.current_position = (x, y)

        return self.current_position

    def update_size(self, new_size):

        self.size = max(3, min(35, int(new_size)))

    def reset_position(self):

        self.previous_position = None
        self.current_position = None
