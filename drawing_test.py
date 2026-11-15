
import cv2
import time
from collections import Counter, deque

from hand_tracking.hand_detector import HandDetector
from gesture_recognition.gesture_detector import GestureDetector

from drawing.canvas import DrawingCanvas
from drawing.brush import Brush
from drawing.drawing_tools import COLORS, DEFAULT_COLOR


class GestureStabilizer:
    def __init__(self, window_size=5, required_votes=3):
        self.history = deque(maxlen=window_size)
        self.required_votes = required_votes
        self.current_gesture = "UNKNOWN"

    def update(self, new_gesture):
        self.history.append(new_gesture)

        if not self.history:
            return self.current_gesture

        counts = Counter(self.history)
        most_common, votes = counts.most_common(1)[0]

        if votes >= self.required_votes:
            self.current_gesture = most_common

        return self.current_gesture

    def reset(self):
        self.history.clear()
        self.current_gesture = "UNKNOWN"


def main():
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        print("ERROR: Cannot access webcam.")
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    hand_detector = None

    try:
        hand_detector = HandDetector(max_hands=1)
        gesture_detector = GestureDetector()

        gesture_stabilizer = GestureStabilizer(
            window_size=5,
            required_votes=3
        )

        canvas = DrawingCanvas(1280, 720)
        brush = Brush(smoothing=0.40)

        current_color = DEFAULT_COLOR
        canvas.set_color(COLORS[current_color])

        color_names = list(COLORS.keys())
        color_index = color_names.index(current_color)

        previous_time = 0.0

        # Drawing grace period
        last_raw_pointing_time = 0.0
        DRAW_GRACE_PERIOD = 0.12

        # Brush size controls
        BRUSH_MIN = 3
        BRUSH_MAX = 35
        BRUSH_STEP = 2
        BRUSH_ADJUST_INTERVAL = 0.30
        last_brush_adjustment = 0.0

        # Colour switching delay
        last_color_change = 0.0
        COLOR_CHANGE_DELAY = 0.8

        print("\n================================")
        print("       MOTION SENSE")
        print("       VIRTUAL CANVAS")
        print("================================")
        print("POINTING    : Draw")
        print("THUMBS UP   : Increase brush size")
        print("THUMBS DOWN : Decrease brush size")
        print("OPEN PALM   : Change colour")
        print("FIST        : Stop drawing")
        print("E           : Toggle eraser")
        print("C           : Clear canvas")
        print("S           : Save artwork")
        print("1-9         : Select colour")
        print("Q           : Exit")
        print("================================\n")

        while True:
            success, frame = camera.read()

            if not success:
                print("Unable to read webcam frame.")
                break

            frame = cv2.resize(frame, (1280, 720))
            frame = cv2.flip(frame, 1)

            height, width, _ = frame.shape
            timestamp = time.monotonic_ns() // 1_000_000

            result = hand_detector.detect(frame, timestamp)
            hand_count = len(result.hand_landmarks)

            gesture = "NO HAND"
            raw_gesture = "NO HAND"

            if hand_count > 0:
                landmarks = result.hand_landmarks[0]

                # Gesture recognition
                raw_gesture = gesture_detector.detect_gesture(landmarks)
                gesture = gesture_stabilizer.update(raw_gesture)

                now = time.monotonic()

                # Index fingertip position
                index_tip = landmarks[8]
                x = int(index_tip.x * width)
                y = int(index_tip.y * height)

                x = max(0, min(width - 1, x))
                y = max(0, min(height - 1, y))

                # ==================================
                # GESTURE 1: POINTING - DRAW
                # ==================================

                if raw_gesture == "POINTING":
                    last_raw_pointing_time = now

                should_draw = (
                    gesture == "POINTING"
                    and (
                        raw_gesture == "POINTING"
                        or (
                            raw_gesture == "UNKNOWN"
                            and now - last_raw_pointing_time <= DRAW_GRACE_PERIOD
                        )
                    )
                )

                if should_draw:
                    smooth_point = brush.smooth_position((x, y))
                    canvas.draw(smooth_point)

                    cv2.circle(
                        frame,
                        smooth_point,
                        max(2, brush.size // 2),
                        COLORS[current_color],
                        -1
                    )

                # ==================================
                # GESTURE 2: THUMBS UP - INCREASE
                # ==================================

                elif (
                    raw_gesture == "THUMBS_UP"
                    and gesture == "THUMBS_UP"
                ):
                    brush.reset_position()
                    canvas.stop_drawing()

                    if now - last_brush_adjustment >= BRUSH_ADJUST_INTERVAL:
                        new_size = min(BRUSH_MAX, brush.size + BRUSH_STEP)
                        brush.update_size(new_size)
                        canvas.set_brush_size(brush.size)
                        last_brush_adjustment = now

                # ==================================
                # GESTURE 3: THUMBS DOWN - DECREASE
                # ==================================

                elif (
                    raw_gesture == "THUMBS_DOWN"
                    and gesture == "THUMBS_DOWN"
                ):
                    brush.reset_position()
                    canvas.stop_drawing()

                    if now - last_brush_adjustment >= BRUSH_ADJUST_INTERVAL:
                        new_size = max(BRUSH_MIN, brush.size - BRUSH_STEP)
                        brush.update_size(new_size)
                        canvas.set_brush_size(brush.size)
                        last_brush_adjustment = now

                # ==================================
                # GESTURE 4: OPEN PALM - CHANGE COLOR
                # ==================================

                elif gesture == "OPEN PALM" and raw_gesture == "OPEN PALM":
                    brush.reset_position()
                    canvas.stop_drawing()

                    if now - last_color_change >= COLOR_CHANGE_DELAY:
                        color_index = (color_index + 1) % len(color_names)
                        current_color = color_names[color_index]
                        canvas.set_color(COLORS[current_color])
                        last_color_change = now

                # ==================================
                # GESTURE 5: FIST - STOP DRAWING
                # ==================================

                elif raw_gesture == "FIST" or gesture == "FIST":
                    brush.reset_position()
                    canvas.stop_drawing()

                # UNKNOWN / TRANSITION
                else:
                    brush.reset_position()
                    canvas.stop_drawing()

                frame = hand_detector.draw_landmarks(frame, result)

            else:
                gesture_stabilizer.reset()
                last_raw_pointing_time = 0.0
                brush.reset_position()
                canvas.stop_drawing()

            # ==================================
            # DISPLAY DRAWING CANVAS
            # ==================================

            drawing_layer = canvas.get_canvas()

            display = cv2.addWeighted(
                frame,
                1.0,
                drawing_layer,
                1.0,
                0
            )

            # FPS
            current_time = time.monotonic()

            fps = (
                1 / (current_time - previous_time)
                if previous_time
                else 0
            )

            previous_time = current_time

            # ==================================
            # DISPLAY INFORMATION
            # ==================================

            cv2.putText(
                display,
                "MOTION SENSE",
                (25, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2
            )

            cv2.putText(
                display,
                f"GESTURE: {gesture}",
                (25, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display,
                f"RAW: {raw_gesture}",
                (25, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                2
            )

            cv2.putText(
                display,
                f"COLOR: {current_color}",
                (25, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                COLORS[current_color],
                2
            )

            cv2.putText(
                display,
                f"BRUSH SIZE: {brush.size}",
                (25, 185),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display,
                f"FPS: {fps:.1f}",
                (25, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                display,
                "Q: Exit | C: Clear | S: Save | E: Eraser",
                (25, height - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1
            )

            cv2.imshow(
                "Motion Sense - Virtual Canvas",
                display
            )

            # ==================================
            # KEYBOARD CONTROLS
            # ==================================

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('c'):
                canvas.clear()
                brush.reset_position()
                canvas.stop_drawing()

            elif key == ord('s'):
                canvas.save()

            elif key == ord('e'):
                canvas.toggle_eraser()

            elif key in [ord(str(i)) for i in range(1, 10)]:
                selected_index = key - ord('1')

                if selected_index < len(color_names):
                    color_index = selected_index
                    current_color = color_names[color_index]
                    canvas.set_color(COLORS[current_color])

        print("Motion Sense session ended.")

    except Exception as error:
        print("Drawing application error:", error)

    finally:
        camera.release()

        if hand_detector is not None:
            hand_detector.close()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
