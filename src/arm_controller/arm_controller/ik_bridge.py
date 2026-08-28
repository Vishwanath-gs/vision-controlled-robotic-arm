import sys
import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from .ik_numeric import numerical_ik


class IKBridge(Node):

    def __init__(self):
        super().__init__('ik_bridge')

        # Send calculated joint angles
        # to the existing arm_brain
        self.angle_pub = self.create_publisher(
            Float64MultiArray,
            '/arm/target_angles',
            10
        )

        # Receive XYZ target
        self.target_sub = self.create_subscription(
            Float64MultiArray,
            '/arm/target_position',
            self.target_callback,
            10
        )

        # Current known joint position
        self.current = [0.0, 0.0, 0.0]

        self.get_logger().info(
            'IK Bridge started. Waiting for XYZ target...'
        )

    def target_callback(self, msg):

        if len(msg.data) != 3:
            self.get_logger().warn(
                'Expected [X, Y, Z]'
            )
            return

        target = (
            msg.data[0],
            msg.data[1],
            msg.data[2]
        )

        self.get_logger().info(
            f'Target XYZ: '
            f'({target[0]:.3f}, '
            f'{target[1]:.3f}, '
            f'{target[2]:.3f})'
        )

        # Calculate IK
        solution, error = numerical_ik(
            target,
            current=tuple(self.current)
        )

        if solution is None:
            self.get_logger().warn(
                'IK could not find a solution.'
            )
            return

        # Check accuracy
        if error > 0.001:
            self.get_logger().warn(
                f'IK error too large: '
                f'{error * 1000:.2f} mm'
            )
            return

        j1, j2, j3 = solution

        # J4 stays at zero for now
        j4 = 0.0

        self.get_logger().info(
            f'IK solution: '
            f'J1={j1:.2f}°, '
            f'J2={j2:.2f}°, '
            f'J3={j3:.2f}°, '
            f'J4={j4:.2f}°'
        )

        # Convert to Float64MultiArray
        angle_msg = Float64MultiArray()

        angle_msg.data = [
            j1,
            j2,
            j3,
            j4
        ]

        # Send angles to existing arm_brain
        self.angle_pub.publish(angle_msg)

        # Remember this as our current commanded pose
        self.current = [
            j1,
            j2,
            j3
        ]


def main(args=None):

    rclpy.init(args=args)

    node = IKBridge()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()