
import cv2
import numpy as np

from pathlib import Path
from datetime import datetime


class DrawingCanvas:

    def __init__(self, width=1280, height=720):

        self.width = width
        self.height = height

        # Create a blank black canvas
        self.canvas = np.zeros(
            (height, width, 3),
            dtype=np.uint8
        )

        self.previous_point = None

        self.color = (255, 255, 0)
        self.brush_size = 10

        self.eraser = False

        print("Drawing canvas initialized!")

    def set_color(self, color):

        self.color = color

        # Selecting a colour exits eraser mode
        self.eraser = False

    def set_brush_size(self, size):

        self.brush_size = max(
            3, min(35, int(size))
        )

    def start_drawing(self, point):

        x, y = point

        self.previous_point = (x, y)

    def draw(self, point):

        x, y = point

        if self.previous_point is None:

            self.previous_point = (x, y)
            return

        color = (
            (0, 0, 0)
            if self.eraser
            else self.color
        )

        cv2.line(
            self.canvas,
            self.previous_point,
            (x, y),
            color,
            self.brush_size,
            cv2.LINE_AA
        )

        self.previous_point = (x, y)

    def stop_drawing(self):

        self.previous_point = None

    def clear(self):

        self.canvas = np.zeros(
            (self.height, self.width, 3),
            dtype=np.uint8
        )

        self.stop_drawing()

        print("Canvas cleared!")

    def toggle_eraser(self):

        self.eraser = not self.eraser

        self.stop_drawing()

        print(
            "Eraser ON" if self.eraser else "Eraser OFF"
        )

    def get_canvas(self):

        return self.canvas.copy()

    def save(self):

        output_folder = (
            Path(__file__).resolve().parents[1]
            / "output"
            / "saved_artwork"
        )

        output_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        filename = (
            "MotionSense_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
            + ".png"
        )

        save_path = output_folder / filename

        success = cv2.imwrite(
            str(save_path),
            self.canvas
        )

        if success:
            print(f"Artwork saved: {save_path}")
        else:
            print("Could not save artwork.")

        return success
