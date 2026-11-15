
import math


class GestureDetector:

    def __init__(self):

        self.PINCH_THRESHOLD = 0.05

        print("Gesture recognition engine initialized!")

    def calculate_distance(self, point1, point2):

        dx = point1.x - point2.x
        dy = point1.y - point2.y

        return math.sqrt(dx**2 + dy**2)

    def is_finger_extended(self, landmarks, tip, pip):

        return landmarks[tip].y < landmarks[pip].y

    def detect_pinch(self, landmarks):

        thumb_tip = landmarks[4]
        index_tip = landmarks[8]

        # Distance between thumb and index fingertip
        pinch_distance = self.calculate_distance(
            thumb_tip, index_tip
        )

        # Measure palm width for scale normalisation
        palm_width = self.calculate_distance(
            landmarks[5], landmarks[17]
        )

        if palm_width == 0:
            return False

        # Relative pinch distance
        pinch_ratio = pinch_distance / palm_width

        return pinch_ratio < 0.55

    def detect_gesture(self, landmarks):

        index_up = self.is_finger_extended(
            landmarks, 8, 6
        )

        middle_up = self.is_finger_extended(
            landmarks, 12, 10
        )

        ring_up = self.is_finger_extended(
            landmarks, 16, 14
        )

        pinky_up = self.is_finger_extended(
            landmarks, 20, 18
        )

        # Gesture 1: Pinch
        if self.detect_pinch(landmarks):
            return "PINCH"

        # Gesture 2: Open palm
        if all([
            index_up,
            middle_up,
            ring_up,
            pinky_up
        ]):
            return "OPEN PALM"

        # Gesture 3: Index finger pointing
        if (
            index_up
            and not middle_up
            and not ring_up
            and not pinky_up
        ):
            return "POINTING"

        # Gesture 4: Closed fist
        if not any([
            index_up,
            middle_up,
            ring_up,
            pinky_up
        ]):
            return "FIST"

        return "UNKNOWN"
