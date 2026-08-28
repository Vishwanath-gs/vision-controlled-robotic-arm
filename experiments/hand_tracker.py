import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera")
    exit()

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as hands:

    while True:
        ret, frame = cap.read()

        if not ret:
            print("ERROR: Could not read camera")
            break

        # Mirror the camera for natural hand movement
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:

            hand = result.multi_hand_landmarks[0]

            # MediaPipe landmark 8 = index fingertip
            fingertip = hand.landmark[8]

            h, w, _ = frame.shape

            px = int(fingertip.x * w)
            py = int(fingertip.y * h)

            # Draw hand skeleton
            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            # Highlight index fingertip
            cv2.circle(
                frame,
                (px, py),
                10,
                (0, 0, 255),
                -1
            )

            # Display normalized coordinates
            text = f"Index: X={fingertip.x:.3f} Y={fingertip.y:.3f}"

            cv2.putText(
                frame,
                text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            print(
                f"Index: "
                f"X={fingertip.x:.3f}, "
                f"Y={fingertip.y:.3f}, "
                f"Z={fingertip.z:.3f}"
            )

        cv2.imshow("Robot Arm Hand Tracker", frame)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
