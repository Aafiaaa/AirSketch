
import cv2
import mediapipe as mp
import time

from pathlib import Path


class HandDetector:

    def __init__(self, max_hands=2):

        # Locate the pretrained model
        project_root = Path(__file__).resolve().parents[1]

        model_path = (
            project_root
            / "assets"
            / "models"
            / "hand_landmarker.task"
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Hand tracking model not found: {model_path}"
            )

        # Configure MediaPipe
        base_options = mp.tasks.BaseOptions(
            model_asset_path=str(model_path)
        )

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Initialize hand detector
        self.detector = (
            mp.tasks.vision.HandLandmarker
            .create_from_options(options)
        )

        print("Hand detector initialized!")

    def detect(self, frame, timestamp_ms):

        # Convert OpenCV BGR image to RGB
        rgb_frame = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )

        # Convert image into MediaPipe format
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Detect hand landmarks
        result = self.detector.detect_for_video(
            mp_image, timestamp_ms
        )

        return result

    def draw_landmarks(self, frame, result):

        height, width, _ = frame.shape

        # Connections between hand landmarks
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        ]

        # Process every detected hand
        for hand_number, hand_landmarks in enumerate(
            result.hand_landmarks
        ):

            points = []

            # Convert normalized coordinates into pixels
            for landmark in hand_landmarks:

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                points.append((x, y))

            # Draw connections
            for start, end in connections:

                cv2.line(
                    frame,
                    points[start],
                    points[end],
                    (0, 255, 0),
                    2
                )

            # Draw landmark points
            for x, y in points:

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 255),
                    -1
                )

            # Highlight important fingertips
            for tip in [4, 8, 12, 16, 20]:

                cv2.circle(
                    frame,
                    points[tip],
                    9,
                    (255, 0, 255),
                    -1
                )

            cv2.putText(
                frame,
                f"Hand {hand_number + 1}",
                points[0],
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        return frame

    def close(self):

        self.detector.close()


# ==========================================
# MOTION SENSE - HAND TRACKING TEST
# ==========================================

if __name__ == "__main__":

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        print("ERROR: Webcam unavailable.")
        raise SystemExit(1)

    detector = None

    try:

        # Initialize hand detector
        detector = HandDetector(max_hands=2)

        print("Motion Sense hand tracking started!")
        print("Show your hand to the camera.")
        print("Press Q to exit.")

        while True:

            success, frame = camera.read()

            if not success:
                print("Unable to read webcam frame.")
                break

            # Mirror webcam feed
            frame = cv2.flip(frame, 1)

            # Generate increasing video timestamps
            timestamp_ms = time.monotonic_ns() // 1_000_000

            # Detect hands
            result = detector.detect(
                frame, timestamp_ms
            )

            # Draw landmarks
            frame = detector.draw_landmarks(
                frame, result
            )

            # Display hand count
            hand_count = len(result.hand_landmarks)

            cv2.putText(
                frame,
                f"Hands Detected: {hand_count}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            cv2.imshow(
                "Motion Sense - Hand Tracking",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as error:
        print("Hand tracking error:", error)

    finally:

        camera.release()

        if detector is not None:
            detector.close()

        cv2.destroyAllWindows()

        print("Hand tracking session ended.")
