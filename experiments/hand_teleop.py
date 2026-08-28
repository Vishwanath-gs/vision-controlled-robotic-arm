import cv2
import mediapipe as mp
import rclpy

from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class HandTeleop(Node):

    def __init__(self):
        super().__init__('hand_teleop')

        self.publisher = self.create_publisher(
            Float64MultiArray,
            '/arm/target_position',
            10
        )

        self.get_logger().info(
            'Hand Teleop started.'
        )


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def main():

    rclpy.init()

    node = HandTeleop()

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(
           "/dev/video0",
           cv2.CAP_V4L2
    )

    cap.set(cv2.CAP_PROP_FOURCC,
             cv2.VideoWriter_fourcc(*"MJPG"))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("ERROR: Could not open camera")
        node.destroy_node()
        rclpy.shutdown()
        return

    # --------------------------------------------------
    # ROBOT WORKSPACE
    # --------------------------------------------------

    # X movement of robot
    ROBOT_X_MIN = -0.07
    ROBOT_X_MAX =  0.07

    # Z movement of robot
    ROBOT_Z_MIN = 0.25
    ROBOT_Z_MAX = 0.35

    # Camera region used for control
    CAMERA_X_MIN = 0.20
    CAMERA_X_MAX = 0.80

    CAMERA_Y_MIN = 0.20
    CAMERA_Y_MAX = 0.80

    # Fixed Y for first experiment
    ROBOT_Y = 0.0

    # Smoothing factor
    alpha = 0.15

    smooth_x = 0.0
    smooth_z = 0.30

    initialized = False

    # --------------------------------------------------

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:

        while rclpy.ok():

            ret, frame = cap.read()

            if not ret:
                print("Camera frame failed")
                break

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            result = hands.process(rgb)

            if result.multi_hand_landmarks:

                hand = result.multi_hand_landmarks[0]

                # Index fingertip
                fingertip = hand.landmark[8]

                camera_x = fingertip.x
                camera_y = fingertip.y

                # --------------------------------------
                # MAP CAMERA X → ROBOT X
                # --------------------------------------

                x_norm = (
                    camera_x - CAMERA_X_MIN
                ) / (
                    CAMERA_X_MAX - CAMERA_X_MIN
                )

                x_norm = clamp(
                    x_norm,
                    0.0,
                    1.0
                )

                target_x = (
                    ROBOT_X_MIN
                    + x_norm *
                    (ROBOT_X_MAX - ROBOT_X_MIN)
                )

                # --------------------------------------
                # MAP CAMERA Y → ROBOT Z
                #
                # Camera Y increases downward.
                # Therefore invert it.
                # --------------------------------------

                y_norm = (
                    camera_y - CAMERA_Y_MIN
                ) / (
                    CAMERA_Y_MAX - CAMERA_Y_MIN
                )

                y_norm = clamp(
                    y_norm,
                    0.0,
                    1.0
                )

                target_z = (
                    ROBOT_Z_MAX
                    - y_norm *
                    (ROBOT_Z_MAX - ROBOT_Z_MIN)
                )

                # --------------------------------------
                # SMOOTH MOVEMENT
                # --------------------------------------

                if not initialized:

                    smooth_x = target_x
                    smooth_z = target_z

                    initialized = True

                else:

                    smooth_x = (
                        alpha * target_x
                        + (1 - alpha) * smooth_x
                    )

                    smooth_z = (
                        alpha * target_z
                        + (1 - alpha) * smooth_z
                    )

                # --------------------------------------
                # SEND XYZ TO EXISTING IK SYSTEM
                # --------------------------------------

                msg = Float64MultiArray()

                msg.data = [
                    float(smooth_x),
                    float(ROBOT_Y),
                    float(smooth_z)
                ]

                node.publisher.publish(msg)

                # --------------------------------------
                # DRAW HAND
                # --------------------------------------

                mp_draw.draw_landmarks(
                    frame,
                    hand,
                    mp_hands.HAND_CONNECTIONS
                )

                h, w, _ = frame.shape

                px = int(
                    fingertip.x * w
                )

                py = int(
                    fingertip.y * h
                )

                cv2.circle(
                    frame,
                    (px, py),
                    10,
                    (0, 0, 255),
                    -1
                )

                # Display robot coordinates

                text1 = (
                    f"Robot X: {smooth_x:.3f} m"
                )

                text2 = (
                    f"Robot Z: {smooth_z:.3f} m"
                )

                cv2.putText(
                    frame,
                    text1,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    text2,
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                # Console output at lower frequency
                node.get_logger().info(
                    f"Target: "
                    f"X={smooth_x:.3f}, "
                    f"Y={ROBOT_Y:.3f}, "
                    f"Z={smooth_z:.3f}"
                )

            cv2.imshow(
                "Robot Arm Hand Teleoperation",
                frame
            )

            # Process ROS events
            rclpy.spin_once(
                node,
                timeout_sec=0.001
            )

            # Q = quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
