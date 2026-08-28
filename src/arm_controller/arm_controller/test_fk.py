import math
from kinematics import forward_kinematics

position = forward_kinematics(
    math.radians(0),
    math.radians(20),
    math.radians(0),
    math.radians(0)
)

print("FK J2=20°:")
print(f"X = {position[0]:.6f} m")
print(f"Y = {position[1]:.6f} m")
print(f"Z = {position[2]:.6f} m")