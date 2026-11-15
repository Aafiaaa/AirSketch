
import cv2
import time

# STEP 1: Initialize webcam
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# STEP 2: Check camera availability
if not camera.isOpened():
    print("ERROR: Webcam could not be accessed.")
    print("Check your camera connection and permissions.")
    exit()

# STEP 3: Set webcam resolution
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Get actual camera resolution
width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("================================")
print("       MOTION SENSE")
print("================================")
print("Webcam initialized successfully!")
print(f"Resolution: {width} x {height}")
print("Press Q to exit.")

# FPS calculation variables
previous_time = 0

# STEP 4: Start video capture loop
try:
    while True:

        # Capture a frame
        success, frame = camera.read()

        if not success:
            print("Unable to capture frame.")
            break

        # STEP 5: Mirror the video
        frame = cv2.flip(frame, 1)

        # STEP 6: Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - previous_time) if previous_time else 0
        previous_time = current_time

        # STEP 7: Add project information
        cv2.putText(
            frame,
            "MOTION SENSE",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "WEBCAM ACTIVE",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (30, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # STEP 8: Display video
        cv2.imshow("Motion Sense - Live Camera", frame)

        # STEP 9: Exit when Q is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # STEP 10: Release webcam and close windows
    camera.release()
    cv2.destroyAllWindows()

    print("Webcam released successfully.")
    print("Motion Sense camera session ended.")
