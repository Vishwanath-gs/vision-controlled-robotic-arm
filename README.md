# Vision-Controlled Robotic Arm 🦾

A ROS 2 and Gazebo based robotic arm controlled using real-time hand
tracking, Cartesian position mapping, inverse kinematics and gesture
control.

## Overview

This project implements vision-based teleoperation of a robotic arm using
a webcam.

A camera tracks the user's hands using MediaPipe. The detected hand
position is converted into a reachable Cartesian target for the robotic
arm. An inverse-kinematics module calculates the required joint angles,
which are then transmitted through ROS 2 to the simulated arm in Gazebo.

## System Architecture

Camera
↓
MediaPipe Hand Tracking
↓
Hand Gesture & Position Detection
↓
Cartesian Target Position
↓
Inverse Kinematics
↓
ROS 2
↓
Gazebo Joint Controllers
↓
Robotic Arm

## Gesture Control

### Right Hand

An open right palm controls the Cartesian workspace.

- Horizontal movement → radial reach
- Vertical movement → height (Z)
- Workspace is constrained to keep targets within the reachable region

### Left Hand

The left hand controls the robot's base rotation.

- Closed hand → J1 rotation control
- Open hand → J1 position locked

## Technologies

- ROS 2 Humble
- Gazebo Sim
- Python
- OpenCV
- MediaPipe
- Inverse Kinematics
- URDF
- ros_gz_sim

## Project Structure

```text
robot_arm_ws/
│
├── src/
│   ├── arm_controller/
│   └── pkg_04_assembly/
│
├── experiments/
│
├── vision_controller.py
├── requirements.txt
├── README.md
└── .gitignore



## Current Features

- Robotic arm modeled using URDF
- Gazebo simulation
- ROS 2 communication between vision, IK and robot control nodes
- Real-time webcam hand tracking using MediaPipe
- Right-hand Cartesian position control
- Reachable workspace mapping with radial distance constraints
- Vertical (Z-axis) position control
- Left-hand gesture-based J1 rotation
- Open-hand J1 locking
- Numerical inverse kinematics
- Joint-limit protection
- Smooth joint trajectory control


## Control Architecture

The robotic arm uses a two-hand interaction model.

**Right Hand — Cartesian Navigation**
- Open palm is used to control the end-effector position.
- Horizontal hand movement controls radial reach.
- Vertical hand movement controls height.
- The target is constrained to the arm's reachable workspace.

**Left Hand — Base Rotation**
- Closed hand activates J1 rotation.
- Horizontal movement rotates the robot around its base.
- Opening the hand locks the current J1 position.

The resulting Cartesian target is passed to the inverse-kinematics
module, which calculates the required joint angles before sending
commands to Gazebo through ROS 2.

## System Pipeline

Webcam
   ↓
MediaPipe Hand Tracking
   ↓
Gesture + Position Detection
   ↓
Reachable Cartesian Target
   ↓
Inverse Kinematics
   ↓
Joint Angles
   ↓
ROS 2
   ↓
Gazebo
   ↓
Robotic Arm


Running the Project
1. Source ROS 2
source /opt/ros/humble/setup.bash
2. Enter the workspace
cd ~/Projects/robot_arm_ws
3. Build
colcon build --symlink-install
4. Source the workspace
source install/setup.bash
5. Start Gazebo
ros2 launch pkg_04_assembly gazebo.launch.py
6. Start the required ROS 2 nodes

Start the ROS-Gazebo bridge, inverse-kinematics bridge and arm controller
according to the project's launch configuration.

7. Start the vision controller

Activate the Python environment:

source vision_env/bin/activate

Then:

python3 vision_controller.py




## Future Improvements

- Gesture-based object grasping and release
- Object detection
- Improved workspace calibration
- Collision avoidance
- Physical robotic-arm implementation

