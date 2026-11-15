
import cv2
import time

from hand_tracking.hand_detector import HandDetector
from gesture_recognition.gesture_detector import GestureDetector


def main():

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        print("Cannot access webcam.")
        return

    hand_detector = HandDetector(max_hands=2)
    gesture_detector = GestureDetector()

    try:

        while True:

            success, frame = camera.read()

            if not success:
                break

            frame = cv2.flip(frame, 1)

            timestamp = time.monotonic_ns() // 1_000_000

            result = hand_detector.detect(
                frame, timestamp
            )

            frame = hand_detector.draw_landmarks(
                frame, result
            )

            # Analyse each detected hand
            for hand in result.hand_landmarks:

                gesture = gesture_detector.detect_gesture(
                    hand
                )

                cv2.putText(
                    frame,
                    gesture,
                    (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2
                )

            cv2.imshow(
                "Motion Sense - Gesture Recognition",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:

        camera.release()
        hand_detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
