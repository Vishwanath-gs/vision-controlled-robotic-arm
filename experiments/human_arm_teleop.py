import cv2
import mediapipe as mp
import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64


# ============================================================
# HELPERS
# ============================================================

def clamp(value, low, high):
    return max(low, min(high, value))


def line_angle(a, b):
    """
    Angle of line A -> B in degrees.
    Image Y is inverted, so we negate dy.
    """
    dx = b[0] - a[0]
    dy = -(b[1] - a[1])

    return math.degrees(math.atan2(dy, dx))


def angle_3points(a, b, c):
    """
    Angle ABC in degrees.
    """
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    mag_ba = math.hypot(*ba)
    mag_bc = math.hypot(*bc)

    if mag_ba < 1e-6 or mag_bc < 1e-6:
        return 0.0

    dot = ba[0] * bc[0] + ba[1] * bc[1]

    value = dot / (mag_ba * mag_bc)
    value = clamp(value, -1.0, 1.0)

    return math.degrees(math.acos(value))


# ============================================================
# ROS NODE
# ============================================================

class HumanArmTeleop(Node):

    def __init__(self):

        super().__init__('human_arm_teleop')

        self.j1_pub = self.create_publisher(
            Float64,
            '/arm/j1/cmd',
            10
        )

        self.j2_pub = self.create_publisher(
            Float64,
            '/arm/j2/cmd',
            10
        )

        self.j3_pub = self.create_publisher(
            Float64,
            '/arm/j3/cmd',
            10
        )

        self.get_logger().info(
            'Human arm teleoperation started'
        )


    def publish_joints(self, j1, j2, j3):

        msg1 = Float64()
        msg2 = Float64()
        msg3 = Float64()

        msg1.data = math.radians(j1)
        msg2.data = math.radians(j2)
        msg3.data = math.radians(j3)

        self.j1_pub.publish(msg1)
        self.j2_pub.publish(msg2)
        self.j3_pub.publish(msg3)


# ============================================================
# MAIN
# ============================================================

def main():

    rclpy.init()

    node = HumanArmTeleop()

    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print("ERROR: Camera could not be opened.")

        node.destroy_node()
        rclpy.shutdown()

        return


    # --------------------------------------------------------
    # ROBOT JOINT LIMITS
    # --------------------------------------------------------

    J1_MIN = -180.0
    J1_MAX = 180.0

    J2_MIN = -104.9
    J2_MAX = 107.1

    J3_MIN = -136.5
    J3_MAX = 135.5


    # --------------------------------------------------------
    # STARTING ROBOT CONFIGURATION
    #
    # This keeps the robot near a known safe pose.
    # --------------------------------------------------------

    target_j1 = 0.0
    target_j2 = 20.0
    target_j3 = 0.0


    # Smoothed values

    smooth_j1 = target_j1
    smooth_j2 = target_j2
    smooth_j3 = target_j3


    # Smoothing strength
    alpha = 0.12


    # --------------------------------------------------------
    # HUMAN ARM TRACKING
    # --------------------------------------------------------

    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as pose:

        while rclpy.ok():

            ret, frame = cap.read()

            if not ret:

                print("Camera frame failed.")
                break


            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            result = pose.process(rgb)


            # =================================================
            # TRACKING FOUND
            # =================================================

            if result.pose_landmarks:

                landmarks = result.pose_landmarks.landmark


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


                # =================================================
                # HUMAN ARM ANGLES
                # =================================================

                upper_angle = line_angle(
                    shoulder_xy,
                    elbow_xy
                )

                elbow_angle = angle_3points(
                    shoulder_xy,
                    elbow_xy,
                    wrist_xy
                )


                # =================================================
                # MAP HUMAN → ROBOT
                # =================================================

                # -------------------------------------------------
                # J1
                #
                # Horizontal arm direction controls robot base.
                #
                # Human arm angle:
                #   -180 ... +180
                #
                # Robot J1:
                #   -90 ... +90
                #
                # We intentionally reduce the range.
                # -------------------------------------------------

                j1_command = clamp(
                    upper_angle,
                    -90.0,
                    90.0
                )


                # -------------------------------------------------
                # J2
                #
                # Human upper-arm elevation.
                #
                # We convert:
                #
                # arm down  ≈ -90
                # horizontal ≈ 0
                # arm up    ≈ +90
                #
                # to robot J2.
                # -------------------------------------------------

                j2_command = clamp(
                    upper_angle,
                    J2_MIN,
                    J2_MAX
                )


                # -------------------------------------------------
                # J3
                #
                # Straight human elbow ≈ 180
                # Bent elbow ≈ 90
                #
                # Convert bend to robot joint movement.
                # -------------------------------------------------

                j3_command = (
                    180.0 - elbow_angle
                )

                j3_command = clamp(
                    j3_command,
                    J3_MIN,
                    J3_MAX
                )


                # =================================================
                # SMOOTHING
                # =================================================

                smooth_j1 = (
                    alpha * j1_command
                    + (1.0 - alpha) * smooth_j1
                )

                smooth_j2 = (
                    alpha * j2_command
                    + (1.0 - alpha) * smooth_j2
                )

                smooth_j3 = (
                    alpha * j3_command
                    + (1.0 - alpha) * smooth_j3
                )


                # =================================================
                # SAFETY CLAMP
                # =================================================

                smooth_j1 = clamp(
                    smooth_j1,
                    J1_MIN,
                    J1_MAX
                )

                smooth_j2 = clamp(
                    smooth_j2,
                    J2_MIN,
                    J2_MAX
                )

                smooth_j3 = clamp(
                    smooth_j3,
                    J3_MIN,
                    J3_MAX
                )


                # =================================================
                # SEND TO ROBOT
                # =================================================

# TEMPORARY:
# Do NOT send commands to robot yet.

                print(
                    f"Human upper={upper_angle:.1f} "
                    f"elbow={elbow_angle:.1f} | "
                    f"Robot J1={smooth_j1:.1f} "
                    f"J2={smooth_j2:.1f} "
                    f"J3={smooth_j3:.1f}"
                )               


                # =================================================
                # DRAW TRACKING
                # =================================================

                mp_draw.draw_landmarks(
                    frame,
                    result.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )


                cv2.circle(
                    frame,
                    shoulder_xy,
                    8,
                    (255, 0, 0),
                    -1
                )

                cv2.circle(
                    frame,
                    elbow_xy,
                    8,
                    (0, 255, 0),
                    -1
                )

                cv2.circle(
                    frame,
                    wrist_xy,
                    8,
                    (0, 0, 255),
                    -1
                )


                # =================================================
                # DISPLAY
                # =================================================

                cv2.putText(
                    frame,
                    f"J1: {smooth_j1:6.1f} deg",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"J2: {smooth_j2:6.1f} deg",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"J3: {smooth_j3:6.1f} deg",
                    (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "RIGHT ARM TELEOPERATION",
                    (20, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "Q = STOP",
                    (20, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )


            # =================================================
            # TRACKING LOST
            # =================================================

            else:

                cv2.putText(
                    frame,
                    "TRACKING LOST - HOLD",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    3
                )


            cv2.imshow(
                "Human Arm -> Robot Arm",
                frame
            )


            rclpy.spin_once(
                node,
                timeout_sec=0.001
            )


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    cap.release()
    cv2.destroyAllWindows()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
