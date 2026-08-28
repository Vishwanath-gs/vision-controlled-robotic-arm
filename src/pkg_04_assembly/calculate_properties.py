import trimesh
import os

DENSITY = 2700.0  # kg/m^3

mesh_dir = os.path.expanduser(
    "~/Downloads/robot_arm_ws/src/pkg_04_assembly/meshes"
)

for i in range(1, 6):
    filename = os.path.join(mesh_dir, f"Part_{i}.stl")

    mesh = trimesh.load_mesh(filename)

    # STL is in mm, convert to metres
    mesh.apply_scale(0.001)

    volume = abs(mesh.volume)
    mass = volume * DENSITY

    print("\n" + "=" * 50)
    print(f"Part {i}")
    print("=" * 50)

    print(f"Volume       : {volume:.9e} m^3")
    print(f"Mass         : {mass:.6f} kg")
    print(f"Mass         : {mass * 1000:.2f} g")

    print("Center of mass [m]:")
    print(mesh.center_mass)

    print("\nInertia tensor about center of mass [kg m^2]:")
    print(mesh.moment_inertia)
    