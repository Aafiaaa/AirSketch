
import math


class GestureDetector:

    def __init__(self):

        # Normalized distance thresholds for two-finger control
        self.CLOSE_THRESHOLD = 0.35
        self.FAR_THRESHOLD = 0.55

        print("Gesture recognition engine initialized!")

    def calculate_distance(self, point1, point2):

        dx = point1.x - point2.x
        dy = point1.y - point2.y

        return math.sqrt(dx**2 + dy**2)

    def is_finger_extended(self, landmarks, tip, pip):

        return landmarks[tip].y < landmarks[pip].y

    def is_thumb_extended_for_open_palm(self, landmarks):

        thumb_tip = landmarks[4]
        index_mcp = landmarks[5]

        palm_width = self.calculate_distance(
            landmarks[5], landmarks[17]
        )

        if palm_width == 0:
            return False

        thumb_distance = self.calculate_distance(
            thumb_tip, index_mcp
        )

        return thumb_distance > palm_width * 0.8

    def detect_size_gesture(self, landmarks):

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

        # Exactly the index and middle fingers must be extended.
        if not (
            index_up
            and middle_up
            and not ring_up
            and not pinky_up
        ):
            return None

        index_tip = landmarks[8]
        middle_tip = landmarks[12]

        finger_distance = self.calculate_distance(
            index_tip, middle_tip
        )

        palm_width = self.calculate_distance(
            landmarks[5], landmarks[17]
        )

        if palm_width == 0:
            return None

        # Normalize for different hand sizes and camera distances.
        distance_ratio = finger_distance / palm_width

        if distance_ratio <= self.CLOSE_THRESHOLD:
            return "SIZE_INCREASE"

        if distance_ratio >= self.FAR_THRESHOLD:
            return "SIZE_DECREASE"

        # Neutral zone: no brush adjustment.
        return "SIZE_NEUTRAL"

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

        four_fingers_up = all([
            index_up,
            middle_up,
            ring_up,
            pinky_up
        ])

        # Gesture 1: Open palm - all five fingers extended.
        if (
            four_fingers_up
            and self.is_thumb_extended_for_open_palm(landmarks)
        ):
            return "OPEN PALM"

        # Gesture 2: Pointing.
        if (
            index_up
            and not middle_up
            and not ring_up
            and not pinky_up
        ):
            return "POINTING"

        # Gesture 3: Two-finger brush size control.
        size_gesture = self.detect_size_gesture(landmarks)

        if size_gesture is not None:
            return size_gesture

        # Gesture 4: Fist.
        if not any([
            index_up,
            middle_up,
            ring_up,
            pinky_up
        ]):
            return "FIST"

        return "UNKNOWN"
