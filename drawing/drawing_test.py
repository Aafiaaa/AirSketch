
import cv2
import time

from hand_tracking.hand_detector import HandDetector
from gesture_recognition.gesture_detector import GestureDetector

from drawing.canvas import DrawingCanvas
from drawing.brush import Brush
from drawing.drawing_tools import COLORS, DEFAULT_COLOR


def main():

    # Initialize webcam
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        print("ERROR: Cannot access webcam.")
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Initialize project components
    hand_detector = HandDetector(max_hands=1)
    gesture_detector = GestureDetector()

    canvas = DrawingCanvas(1280, 720)
    brush = Brush()

    current_color = DEFAULT_COLOR
    canvas.set_color(COLORS[current_color])

    color_names = list(COLORS.keys())
    color_index = color_names.index(current_color)

    previous_time = 0
    last_color_change = 0

    print("MOTION SENSE - VIRTUAL CANVAS")
    print("--------------------------------")
    print("POINTING : Draw")
    print("PINCH    : Adjust brush size")
    print("OPEN PALM: Change colour")
    print("FIST     : Stop drawing")
    print("E        : Toggle eraser")
    print("C        : Clear canvas")
    print("S        : Save artwork")
    print("Q        : Exit")

    try:

        while True:

            success, frame = camera.read()

            if not success:
                break

            # Mirror camera
            frame = cv2.resize(frame, (1280, 720))
            frame = cv2.flip(frame, 1)

            height, width, _ = frame.shape

            # Detect hand
            timestamp = time.monotonic_ns() // 1_000_000

            result = hand_detector.detect(
                frame, timestamp
            )

            hand_count = len(result.hand_landmarks)

            gesture = "NO HAND"

            if hand_count > 0:

                landmarks = result.hand_landmarks[0]

                gesture = gesture_detector.detect_gesture(
                    landmarks
                )

                # Get index fingertip coordinates
                index_tip = landmarks[8]

                x = int(index_tip.x * width)
                y = int(index_tip.y * height)

                # Clamp coordinates to canvas bounds
                x = max(0, min(width - 1, x))
                y = max(0, min(height - 1, y))

                # POINTING: Draw
                if gesture == "POINTING":

                    smooth_point = brush.smooth_position(
                        (x, y)
                    )

                    canvas.draw(smooth_point)

                    cv2.circle(
                        frame,
                        smooth_point,
                        brush.size // 2,
                        COLORS[current_color],
                        -1
                    )

                # PINCH: Adjust brush size
                elif gesture == "PINCH":

                    brush.reset_position()
                    canvas.stop_drawing()

                    distance = gesture_detector.calculate_distance(
                        landmarks[4],
                        landmarks[8]
                    )

                    # Map pinch distance to brush size
                    new_size = 3 + (
                        distance / 0.05
                    ) * 32

                    brush.update_size(new_size)
                    canvas.set_brush_size(brush.size)

                # OPEN PALM: Change colour
                elif gesture == "OPEN PALM":

                    brush.reset_position()
                    canvas.stop_drawing()

                    current_time = time.time()

                    if current_time - last_color_change > 0.8:

                        color_index = (
                            color_index + 1
                        ) % len(color_names)

                        current_color = color_names[color_index]

                        canvas.set_color(
                            COLORS[current_color]
                        )

                        last_color_change = current_time

                # FIST: Pause drawing
                elif gesture == "FIST":

                    brush.reset_position()
                    canvas.stop_drawing()

                else:

                    brush.reset_position()
                    canvas.stop_drawing()

                # Draw hand landmarks
                frame = hand_detector.draw_landmarks(
                    frame, result
                )

            else:

                brush.reset_position()
                canvas.stop_drawing()

            # Get drawing canvas
            drawing_layer = canvas.get_canvas()

            # Overlay artwork on camera feed
            display = cv2.addWeighted(
                frame,
                1.0,
                drawing_layer,
                1.0,
                0
            )

            # Calculate FPS
            current_time = time.time()

            fps = (
                1 / (current_time - previous_time)
                if previous_time
                else 0
            )

            previous_time = current_time

            # Display information
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
                f"COLOR: {current_color}",
                (25, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                COLORS[current_color],
                2
            )

            cv2.putText(
                display,
                f"BRUSH SIZE: {brush.size}",
                (25, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display,
                f"FPS: {fps:.1f}",
                (25, 185),
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

            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('c'):
                canvas.clear()

            elif key == ord('s'):
                canvas.save()

            elif key == ord('e'):
                canvas.toggle_eraser()

            elif key in [ord(str(i)) for i in range(1, 10)]:

                selected_index = key - ord('1')

                if selected_index < len(color_names):

                    color_index = selected_index
                    current_color = color_names[color_index]

                    canvas.set_color(
                        COLORS[current_color]
                    )

    except Exception as error:

        print("Drawing application error:", error)

    finally:

        camera.release()
        hand_detector.close()
        cv2.destroyAllWindows()

        print("Motion Sense drawing session ended.")


if __name__ == "__main__":
    main()
