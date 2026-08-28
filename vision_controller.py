import math
import time

import cv2
import mediapipe as mp
import rclpy

from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


# ============================================================
# ROBOT WORKSPACE
# ============================================================

# IMPORTANT:
# These values describe RADIAL distance from the robot base,
# NOT raw X coordinates.
#
# We deliberately keep a safe minimum radius because the
# numerical IK becomes unreliable very close to the base.

R_MIN = 0.085
R_MAX = 0.145

# Vertical workspace.
Z_MIN = 0.270
Z_MAX = 0.350


# ============================================================
# J1 SETTINGS
# ============================================================

J1_MIN = math.radians(-80.0)
J1_MAX = math.radians(80.0)

# Horizontal movement of the LEFT CLOSED hand controls J1.
J1_GAIN = math.radians(180.0)


# ============================================================
# SMOOTHING
# ============================================================

POSITION_ALPHA = 0.20
J1_ALPHA = 0.20

# Ignore tiny camera noise.
R_DEADBAND = 0.003
Z_DEADBAND = 0.003
J1_DEADBAND = 0.015

# Vision target publication rate.
PUBLISH_PERIOD = 0.10


# ============================================================
# HELPERS
# ============================================================

def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def distance(a, b):
    return math.hypot(
        a.x - b.x,
        a.y - b.y
    )


def finger_extended(landmarks, tip, pip):
    wrist = landmarks[0]

    return (
        distance(landmarks[tip], wrist)
        >
        distance(landmarks[pip], wrist) * 1.10
    )


def classify_hand(hand):
    """
    OPEN:
        3 or more fingers extended

    CLOSED:
        fewer than 3 fingers extended
    """

    lm = hand.landmark

    fingers = [
        finger_extended(lm, 8, 6),    # index
        finger_extended(lm, 12, 10),  # middle
        finger_extended(lm, 16, 14),  # ring
        finger_extended(lm, 20, 18),  # little
    ]

    return "OPEN" if sum(fingers) >= 3 else "CLOSED"


def palm_center(hand):
    """
    Stable palm position using five landmarks:
        wrist
        index MCP
        middle MCP
        ring MCP
        little MCP
    """

    lm = hand.landmark

    x = (
        lm[0].x +
        lm[5].x +
        lm[9].x +
        lm[13].x +
        lm[17].x
    ) / 5.0

    y = (
        lm[0].y +
        lm[5].y +
        lm[9].y +
        lm[13].y +
        lm[17].y
    ) / 5.0

    return x, y


# ============================================================
# ROS NODE
# ============================================================

class VisionController(Node):

    def __init__(self):

        super().__init__("vision_controller")

        self.target_pub = self.create_publisher(
            Float64MultiArray,
            "/arm/target_position",
            10
        )

        self.get_logger().info(
            "========================================"
        )

        self.get_logger().info(
            "VISION ROBOT CONTROLLER"
        )

        self.get_logger().info(
            "RIGHT OPEN  -> Cartesian movement"
        )

        self.get_logger().info(
            "LEFT CLOSED -> J1 rotation"
        )

        self.get_logger().info(
            "LEFT OPEN   -> J1 lock"
        )

        self.get_logger().info(
            "========================================"
        )

    def publish_target(self, x, y, z):

        msg = Float64MultiArray()

        msg.data = [
            float(x),
            float(y),
            float(z)
        ]

        self.target_pub.publish(msg)


# ============================================================
# MAIN
# ============================================================

def main():

    rclpy.init()

    node = VisionController()

    # ========================================================
    # CAMERA
    # ========================================================

    cap = None

    for camera_index in [0, 1]:

        test_cap = cv2.VideoCapture(camera_index)

        if test_cap.isOpened():

            cap = test_cap

            node.get_logger().info(
                f"Camera opened: /dev/video{camera_index}"
            )

            break

        test_cap.release()

    if cap is None:

        node.get_logger().error(
            "No camera could be opened."
        )

        node.destroy_node()
        rclpy.shutdown()

        return

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        640
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        480
    )

    # ========================================================
    # MEDIAPIPE
    # ========================================================

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    # ========================================================
    # ROBOT STATE
    # ========================================================

    # Start in a known reachable position.

    target_radius = 0.105
    target_z = 0.310

    smooth_radius = target_radius
    smooth_z = target_z

    # J1
    j1 = 0.0
    smooth_j1 = 0.0
    locked_j1 = 0.0

    # Left-hand J1 state
    left_was_closed = False
    left_anchor_x = None

    # Hand status
    right_state = "NOT FOUND"
    left_state = "NOT FOUND"

    # Publish timer
    last_publish = 0.0

    # ========================================================
    # MEDIAPIPE
    # ========================================================

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.65
    ) as hands:

        while rclpy.ok():

            # =================================================
            # CAMERA
            # =================================================

            ret, frame = cap.read()

            if not ret:
                continue

            # Mirror image so movement feels natural.
            frame = cv2.flip(frame, 1)

            # =================================================
            # HAND DETECTION
            # =================================================

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            result = hands.process(rgb)

            right = None
            left = None

            if result.multi_hand_landmarks:

                for hand, handedness in zip(
                    result.multi_hand_landmarks,
                    result.multi_handedness
                ):

                    label = (
                        handedness
                        .classification[0]
                        .label
                    )

                    state = classify_hand(hand)

                    x, y = palm_center(hand)

                    data = {
                        "hand": hand,
                        "state": state,
                        "x": x,
                        "y": y
                    }

                    if label == "Right":
                        right = data

                    elif label == "Left":
                        left = data

                    mp_draw.draw_landmarks(
                        frame,
                        hand,
                        mp_hands.HAND_CONNECTIONS
                    )

            # =================================================
            # RIGHT HAND
            #
            # OPEN RIGHT PALM:
            #
            # Camera X -> radial distance
            # Camera Y -> Z
            #
            # We do NOT map camera X directly to robot X.
            #
            # This prevents the target from getting too close
            # to the robot base.
            # =================================================

            if right is not None:

                right_state = right["state"]

                if right_state == "OPEN":

                    cam_x = clamp(
                        right["x"],
                        0.0,
                        1.0
                    )

                    cam_y = clamp(
                        right["y"],
                        0.0,
                        1.0
                    )

                    # -----------------------------------------
                    # CAMERA X -> RADIAL REACH
                    # -----------------------------------------

                    new_radius = (
                        R_MIN +
                        cam_x * (R_MAX - R_MIN)
                    )

                    new_radius = clamp(
                        new_radius,
                        R_MIN,
                        R_MAX
                    )

                    # -----------------------------------------
                    # CAMERA Y -> HEIGHT
                    # -----------------------------------------

                    new_z = (
                        Z_MIN +
                        (1.0 - cam_y)
                        * (Z_MAX - Z_MIN)
                    )

                    new_z = clamp(
                        new_z,
                        Z_MIN,
                        Z_MAX
                    )

                    # -----------------------------------------
                    # DEAD BAND
                    # -----------------------------------------

                    if abs(
                        new_radius - smooth_radius
                    ) > R_DEADBAND:

                        target_radius = new_radius

                    if abs(
                        new_z - smooth_z
                    ) > Z_DEADBAND:

                        target_z = new_z

                    # -----------------------------------------
                    # SMOOTH
                    # -----------------------------------------

                    smooth_radius = (
                        POSITION_ALPHA * target_radius
                        +
                        (1.0 - POSITION_ALPHA)
                        * smooth_radius
                    )

                    smooth_z = (
                        POSITION_ALPHA * target_z
                        +
                        (1.0 - POSITION_ALPHA)
                        * smooth_z
                    )

            else:

                right_state = "NOT FOUND"

            # =================================================
            # LEFT HAND
            #
            # CLOSED -> J1 CONTROL
            # OPEN   -> J1 LOCK
            # =================================================

            if left is not None:

                left_state = left["state"]

                # ------------------------------------------------
                # CLOSED
                # ------------------------------------------------

                if left_state == "CLOSED":

                    # Detect transition:
                    # OPEN -> CLOSED

                    if not left_was_closed:

                        left_anchor_x = left["x"]

                        locked_j1 = j1

                        left_was_closed = True

                        node.get_logger().info(
                            "J1 CONTROL ACTIVE"
                        )

                    # Move J1 relative to the point where
                    # the fist was closed.

                    if left_anchor_x is not None:

                        dx = (
                            left["x"]
                            -
                            left_anchor_x
                        )

                        if abs(dx) > J1_DEADBAND:

                            requested_j1 = (
                                locked_j1
                                +
                                dx * J1_GAIN
                            )

                            requested_j1 = clamp(
                                requested_j1,
                                J1_MIN,
                                J1_MAX
                            )

                            smooth_j1 = (
                                J1_ALPHA * requested_j1
                                +
                                (1.0 - J1_ALPHA)
                                * smooth_j1
                            )

                            j1 = smooth_j1

                # ------------------------------------------------
                # OPEN
                # ------------------------------------------------

                elif left_state == "OPEN":

                    if left_was_closed:

                        locked_j1 = j1

                        left_anchor_x = None

                        left_was_closed = False

                        node.get_logger().info(
                            f"J1 LOCKED = "
                            f"{math.degrees(j1):.1f} deg"
                        )

            else:

                left_state = "NOT FOUND"

            # =================================================
            # RADIAL -> XYZ
            #
            # J1 rotates the planar workspace.
            #
            # radius + J1 determine X/Y.
            # right hand determines Z.
            # =================================================

            robot_x = (
                smooth_radius *
                math.cos(j1)
            )

            robot_y = (
                smooth_radius *
                math.sin(j1)
            )

            robot_z = clamp(
                smooth_z,
                Z_MIN,
                Z_MAX
            )

            # =================================================
            # PUBLISH TARGET
            # =================================================

            now = time.monotonic()

            if (
                right is not None
                and right_state == "OPEN"
                and
                now - last_publish >= PUBLISH_PERIOD
            ):

                node.publish_target(
                    robot_x,
                    robot_y,
                    robot_z
                )

                last_publish = now

            # =================================================
            # DISPLAY
            # =================================================

            overlay = frame.copy()

            cv2.rectangle(
                overlay,
                (10, 10),
                (630, 185),
                (20, 20, 20),
                -1
            )

            frame = cv2.addWeighted(
                overlay,
                0.65,
                frame,
                0.35,
                0
            )

            cv2.putText(
                frame,
                f"RIGHT: {right_state}",
                (25, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"LEFT:  {left_state}",
                (25, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"J1: {math.degrees(j1):+.1f} deg",
                (25, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"R: {smooth_radius:.3f} m",
                (25, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Z: {robot_z:.3f} m",
                (220, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"X: {robot_x:+.3f}  Y: {robot_y:+.3f}",
                (370, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "RIGHT OPEN=MOVE   LEFT FIST=J1   LEFT OPEN=LOCK",
                (25, 165),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (255, 255, 255),
                1
            )

            cv2.imshow(
                "Robot Arm Vision Controller",
                frame
            )

            # =================================================
            # ROS
            # =================================================

            rclpy.spin_once(
                node,
                timeout_sec=0.001
            )

            # =================================================
            # QUIT
            # =================================================

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()