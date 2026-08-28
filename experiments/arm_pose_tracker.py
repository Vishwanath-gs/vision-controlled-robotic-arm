import cv2
import mediapipe as mp
import math


def angle_3points(a, b, c):
    """Angle ABC in degrees."""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    mag_ba = math.hypot(*ba)
    mag_bc = math.hypot(*bc)

    if mag_ba == 0 or mag_bc == 0:
        return 0.0

    dot = ba[0] * bc[0] + ba[1] * bc[1]

    value = dot / (mag_ba * mag_bc)
    value = max(-1.0, min(1.0, value))

    return math.degrees(math.acos(value))


def line_angle(a, b):
    """Angle of line A -> B relative to horizontal."""
    dx = b[0] - a[0]
    dy = b[1] - a[1]

    return math.degrees(math.atan2(-dy, dx))


mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()


with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as pose, mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as hands:

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Camera frame failed.")
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        pose_result = pose.process(rgb)
        hand_result = hands.process(rgb)

        # ------------------------------------------
        # RIGHT ARM
        # ------------------------------------------

        if pose_result.pose_landmarks:

            landmarks = pose_result.pose_landmarks.landmark

            shoulder = landmarks[
                mp_pose.PoseLandmark.RIGHT_SHOULDER
            ]

            elbow = landmarks[
                mp_pose.PoseLandmark.RIGHT_ELBOW
            ]

            wrist = landmarks[
                mp_pose.PoseLandmark.RIGHT_WRIST
            ]

            h, w, _ = frame.shape

            shoulder_xy = (
                int(shoulder.x * w),
                int(shoulder.y * h)
            )

            elbow_xy = (
                int(elbow.x * w),
                int(elbow.y * h)
            )

            wrist_xy = (
                int(wrist.x * w),
                int(wrist.y * h)
            )

            # --------------------------------------
            # SHOULDER / UPPER ARM ANGLE
            # --------------------------------------

            upper_arm_angle = line_angle(
                shoulder_xy,
                elbow_xy
            )

            # --------------------------------------
            # ELBOW ANGLE
            # --------------------------------------

            elbow_angle = angle_3points(
                shoulder_xy,
                elbow_xy,
                wrist_xy
            )

            # --------------------------------------
            # FOREARM ANGLE
            # --------------------------------------

            forearm_angle = line_angle(
                elbow_xy,
                wrist_xy
            )

            # --------------------------------------
            # DRAW RIGHT ARM
            # --------------------------------------

            cv2.circle(
                frame,
                shoulder_xy,
                9,
                (255, 0, 0),
                -1
            )

            cv2.circle(
                frame,
                elbow_xy,
                9,
                (0, 255, 0),
                -1
            )

            cv2.circle(
                frame,
                wrist_xy,
                9,
                (0, 0, 255),
                -1
            )

            cv2.line(
                frame,
                shoulder_xy,
                elbow_xy,
                (255, 255, 255),
                3
            )

            cv2.line(
                frame,
                elbow_xy,
                wrist_xy,
                (255, 255, 255),
                3
            )

            # --------------------------------------
            # TEXT
            # --------------------------------------

            cv2.putText(
                frame,
                f"Upper arm: {upper_arm_angle:.1f} deg",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Elbow: {elbow_angle:.1f} deg",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Forearm: {forearm_angle:.1f} deg",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )

        # ------------------------------------------
        # RIGHT INDEX FINGER
        # ------------------------------------------

        if hand_result.multi_hand_landmarks:

            # Because we use max_num_hands=1,
            # this is our detected hand.
            hand = hand_result.multi_hand_landmarks[0]

            fingertip = hand.landmark[8]

            px = int(fingertip.x * w)
            py = int(fingertip.y * h)

            cv2.circle(
                frame,
                (px, py),
                12,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                frame,
                "INDEX",
                (px + 15, py),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

        # ------------------------------------------
        # DRAW POSE
        # ------------------------------------------

        if pose_result.pose_landmarks:

            mp_draw.draw_landmarks(
                frame,
                pose_result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        cv2.putText(
            frame,
            "RIGHT ARM TELEOPERATION TRACKER",
            (20, h - 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Q = quit",
            (20, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Human Arm Tracking",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()