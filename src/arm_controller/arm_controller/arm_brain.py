import math

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64
from std_msgs.msg import Float64MultiArray


class ArmBrain(Node):

    def __init__(self):

        super().__init__("arm_brain")

        # ====================================================
        # GAZEBO JOINT COMMANDS
        # ====================================================

        self.j1_pub = self.create_publisher(
            Float64,
            "/arm/j1/cmd",
            10
        )

        self.j2_pub = self.create_publisher(
            Float64,
            "/arm/j2/cmd",
            10
        )

        self.j3_pub = self.create_publisher(
            Float64,
            "/arm/j3/cmd",
            10
        )

        self.j4_pub = self.create_publisher(
            Float64,
            "/arm/j4/cmd",
            10
        )

        # ====================================================
        # IK TARGET
        # ====================================================

        self.target_sub = self.create_subscription(
            Float64MultiArray,
            "/arm/target_angles",
            self.target_callback,
            10
        )

        # ====================================================
        # JOINT LIMITS
        #
        # Stored in RADIANS.
        # ====================================================

        self.limits = {

            "j1": (
                -math.pi,
                math.pi
            ),

            "j2": (
                -1.83088,
                1.86921
            ),

            "j3": (
                -2.38278,
                2.36452
            ),

            "j4": (
                -math.pi,
                math.pi
            )
        }

        # ====================================================
        # CURRENT POSITION
        # ====================================================

        self.current = {
            "j1": 0.0,
            "j2": 0.0,
            "j3": 0.0,
            "j4": 0.0
        }

        # ====================================================
        # TARGET POSITION
        # ====================================================

        self.target = self.current.copy()

        # ====================================================
        # TELEOPERATION SMOOTHING
        #
        # Much faster than the old 2-second trajectory.
        # ====================================================

        self.step_time = 0.02

        self.motion_time = 0.10

        self.steps = max(
            1,
            int(
                self.motion_time /
                self.step_time
            )
        )

        self.step_count = self.steps

        # ====================================================
        # TRAJECTORY START
        # ====================================================

        self.start = self.current.copy()

        # ====================================================
        # TIMER
        # ====================================================

        self.timer = self.create_timer(
            self.step_time,
            self.trajectory_step
        )

        self.get_logger().info(
            "Arm Brain started."
        )

        self.get_logger().info(
            "Waiting for target angles..."
        )

    # ========================================================
    # TARGET CALLBACK
    # ========================================================

    def target_callback(self, msg):

        if len(msg.data) != 4:

            self.get_logger().warn(
                "Expected 4 angles: "
                "[J1, J2, J3, J4]"
            )

            return

        # ====================================================
        # IK OUTPUT IS DEGREES.
        #
        # GAZEBO COMMANDS ARE RADIANS.
        # ====================================================

        requested = {

            "j1": math.radians(
                msg.data[0]
            ),

            "j2": math.radians(
                msg.data[1]
            ),

            "j3": math.radians(
                msg.data[2]
            ),

            "j4": math.radians(
                msg.data[3]
            )
        }

        self.target = self.clamp_targets(
            requested
        )

        # Start a short interpolation from wherever
        # the robot currently is.

        self.start = self.current.copy()

        self.step_count = 0

    # ========================================================
    # CLAMP
    # ========================================================

    def clamp_targets(self, targets):

        safe_targets = {}

        for joint, angle in targets.items():

            minimum, maximum = self.limits[joint]

            safe_angle = max(
                minimum,
                min(angle, maximum)
            )

            if safe_angle != angle:

                self.get_logger().warn(
                    f"{joint} exceeded limit. "
                    f"Clamped to "
                    f"{math.degrees(safe_angle):.1f} deg"
                )

            safe_targets[joint] = safe_angle

        return safe_targets

    # ========================================================
    # TRAJECTORY
    # ========================================================

    def trajectory_step(self):

        if self.step_count >= self.steps:

            # Keep publishing final target.
            self.current = self.target.copy()

            self.publish_current()

            return

        self.step_count += 1

        progress = (
            self.step_count /
            self.steps
        )

        # Smoothstep.
        smooth = (
            progress *
            progress *
            (3.0 - 2.0 * progress)
        )

        for joint in self.current:

            start_angle = self.start[joint]

            target_angle = self.target[joint]

            self.current[joint] = (
                start_angle
                +
                (
                    target_angle -
                    start_angle
                )
                * smooth
            )

        self.publish_current()

    # ========================================================
    # PUBLISH
    # ========================================================

    def publish_current(self):

        messages = {

            "j1": self.j1_pub,

            "j2": self.j2_pub,

            "j3": self.j3_pub,

            "j4": self.j4_pub
        }

        for joint, publisher in messages.items():

            msg = Float64()

            msg.data = float(
                self.current[joint]
            )

            publisher.publish(msg)


# ============================================================
# MAIN
# ============================================================

def main(args=None):

    rclpy.init(args=args)

    node = ArmBrain()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":
    main()