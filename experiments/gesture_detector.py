import cv2
import mediapipe as mp
import math


# ============================================================
# FINGER STATE
# ============================================================

def distance(a, b):
    return math.hypot(
        a.x - b.x,
        a.y - b.y
    )


def finger_extended(landmarks, tip, pip):
    """
    Simple finger-extension test.
    If fingertip is farther from wrist than PIP,
    the finger is considered extended.
    """

    wrist = landmarks[0]

    tip_dist = distance(
        landmarks[tip],
        wrist
    )

    pip_dist = distance(
        landmarks[pip],
        wrist
    )

    return tip_dist > pip_dist * 1.10


def classify_hand(hand_landmarks):

    lm = hand_landmarks.landmark

    # Index
    index = finger_extended(
        lm, 8, 6
    )

    # Middle
    middle = finger_extended(
        lm, 12, 10
    )

    # Ring
    ring = finger_extended(
        lm, 16, 14
    )

    # Pinky
    pinky = finger_extended(
        lm, 20, 18
    )

    extended_count = sum([
        index,
        middle,
        ring,
        pinky
    ])

    # Four fingers extended = OPEN
    if extended_count >= 3:
        return "OPEN"

    # Mostly closed = CLOSED
    return "CLOSED"


# ============================================================
# HAND LABEL
# ============================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Camera could not be opened.")
    exit()


# ============================================================
# MAIN
# ============================================================

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as hands:

    while True:

        ret, frame = cap.read()

        if not ret:

            print("Camera frame failed.")
            break

        # Mirror camera
        frame = cv2.flip(
            frame,
            1
        )

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        result = hands.process(rgb)

        right_state = "NOT FOUND"
        left_state = "NOT FOUND"

        if result.multi_hand_landmarks:

            for hand_landmarks, handedness in zip(
                result.multi_hand_landmarks,
                result.multi_handedness
            ):

                # MediaPipe handedness is based on
                # the mirrored/selfie convention.
                label = handedness.classification[0].label

                state = classify_hand(
                    hand_landmarks
                )

                # ------------------------------------------
                # HAND IDENTITY
                # ------------------------------------------

                if label == "Right":
                    right_state = state
                else:
                    left_state = state

                # ------------------------------------------
                # DRAW HAND
                # ------------------------------------------

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                # Wrist position
                wrist = hand_landmarks.landmark[0]

                h, w, _ = frame.shape

                px = int(wrist.x * w)
                py = int(wrist.y * h)

                cv2.putText(
                    frame,
                    f"{label}: {state}",
                    (px - 60, py - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2
                )

        # ====================================================
        # CONTROL STATE
        # ====================================================

        if right_state == "OPEN":

            right_mode = "NAVIGATE"

        elif right_state == "CLOSED":

            right_mode = "GRAB"

        else:

            right_mode = "HOLD"


        if left_state == "CLOSED":

            left_mode = "ROTATE J1"

        elif left_state == "OPEN":

            left_mode = "LOCK J1"

        else:

            left_mode = "HOLD J1"


        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.rectangle(
            frame,
            (10, 10),
            (620, 145),
            (30, 30, 30),
            -1
        )

        cv2.putText(
            frame,
            f"RIGHT: {right_state}",
            (25, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"RIGHT MODE: {right_mode}",
            (25, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"LEFT:  {left_state}",
            (25, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"LEFT MODE: {left_mode}",
            (25, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Robot Arm Gesture Controller",
            frame
        )

        # Q = quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


cap.release()
cv2.destroyAllWindows()
