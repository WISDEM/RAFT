import math
import pygmsh
import meshmagick
import subprocess
import numpy as np
import os
import platform
import yaml


def load_yaml(file_path):
    try:
        with open(file_path, "r") as file:
            yaml_data = yaml.safe_load(file)
        return yaml_data
    except Exception as e:
        print(f"Error loading YAML file: {e}")
        return None


def meshMember(geom, headings, rA, rB, radius, member_id=0,
               stations=[0.0, 1.0], diameters=None, extensionA=0, extensionB=0):
    cylinders = []

    # Compute full axis and direction unit vector
    axis_vec = [rB[i] - rA[i] for i in range(3)]
    axis_length = math.sqrt(sum(a**2 for a in axis_vec))
    direction = [a / axis_length for a in axis_vec]

    #  Apply extensions
    rA_ext = [rA[i] - extensionA * direction[i] for i in range(3)]
    rB_ext = [rB[i] + extensionB * direction[i] for i in range(3)]
    axis_full = [rB_ext[i] - rA_ext[i] for i in range(3)]

    uniform = isinstance(diameters, (int, float)) or (isinstance(diameters, list) and len(diameters) == 1)

    for idx in range(len(headings)):
        if np.all(diameters==diameters[0]):
            start = rA_ext
            end   = rB_ext
            axis_segment = [end[i] - start[i] for i in range(3)]
            cone = geom.add_cylinder(start, axis_segment, diameters[0]/2)
            label = f"Cylinder_{member_id}_{idx}"
            print(f"Meshing {label} | Start: {start} | End: {end} | Radius: {diameters[0]/2}->{diameters[0]/2}")

            geom.add_physical(cone, label=label)
            cylinders.append(cone)
        else:
            for s in range(len(stations) - 1):
                t0 = (stations[s] - stations[0]) / (stations[-1] - stations[0])
                t1 = (stations[s + 1] - stations[0]) / (stations[-1] - stations[0])

                if abs(t1 - t0) < 1e-6:
                    continue

                # ⏱ Interpolate segment along extended axis
                start = [rA_ext[i] + t0 * axis_full[i] for i in range(3)]
                end   = [rA_ext[i] + t1 * axis_full[i] for i in range(3)]
                axis_segment = [end[i] - start[i] for i in range(3)]

                if uniform or diameters is None:
                    radius_start = radius_end = radius
                else:
                    radius_start = diameters[s] / 2
                    radius_end = diameters[s + 1] / 2

                label = f"Cylinder_{member_id}_{idx}_seg{s}"
                print(f"Meshing {label} | Start: {start} | End: {end} | Radius: {radius_start}->{radius_end}")

                if abs(radius_start - radius_end) < 1e-6:
                    cone = geom.add_cylinder(start, axis_segment, radius_start)
                else:
                    cone = geom.add_cone(start, axis_segment, radius_start, radius_end)

                geom.add_physical(cone, label=label)
                cylinders.append(cone)

    return cylinders



def meshRectangularMember(geom, heading, rA, rB, widths, heights, member_id=0,
                          stations=[0.0, 1.0], extensionA=0, extensionB=0):
    boxes = []

    # Compute axis and unit vector
    axis_vec = [rB[i] - rA[i] for i in range(3)]
    axis_length = math.sqrt(sum(a ** 2 for a in axis_vec))
    direction = [a / axis_length for a in axis_vec]

    # Apply extensions to both ends
    rA_ext = [rA[i] - extensionA * direction[i] for i in range(3)]
    rB_ext = [rB[i] + extensionB * direction[i] for i in range(3)]
    axis_full = [rB_ext[i] - rA_ext[i] for i in range(3)]

    for s in range(len(stations) - 1):
        t0 = (stations[s] - stations[0]) / (stations[-1] - stations[0])
        t1 = (stations[s + 1] - stations[0]) / (stations[-1] - stations[0])

        if abs(t1 - t0) < 1e-6:
            continue

        start = [rA_ext[i] + t0 * axis_full[i] for i in range(3)]
        end = [rA_ext[i] + t1 * axis_full[i] for i in range(3)]
        axis_segment = [end[i] - start[i] for i in range(3)]

        width = widths[s]
        height = heights[s]
        length = math.sqrt(sum(a ** 2 for a in axis_segment))
        box_size = [length, width, height]

        # Box at origin
        box = geom.add_box(
            [-box_size[0] / 2, -box_size[1] / 2, -box_size[2] / 2],
            box_size
        )

        # Rotation to align
        dx, dy = axis_segment[0], axis_segment[1]
        theta = math.atan2(dy, dx)
        geom.rotate(box, point=(0, 0, 0), angle=theta, axis=(0, 0, 1))

        # Translate box to its center
        box_center = [
            start[0] + axis_segment[0] / 2,
            start[1] + axis_segment[1] / 2,
            start[2] + axis_segment[2] / 2
        ]
        geom.translate(box, box_center)

        label = f"Box_{member_id}_seg{s}"
        print(f"Box: {label} | Center: {box_center} | Heading: {math.degrees(theta):.1f} | theta: {theta:.1f} | Size: {box_size}")
        geom.add_physical(box, label=label)
        boxes.append(box)

    return boxes


def mesh(meshDir=os.path.join(os.getcwd(),'BEM'), cylindrical_members=[], rectangular_members=[], dmin=0.1, dmax=1):

    print(f"Total cylindrical members: {len(cylindrical_members)}")
    print(f"Total rectangular members: {len(rectangular_members)}")
    all_shapes = []

    with pygmsh.occ.Geometry() as geom:
        geom.characteristic_length_min = dmin
        geom.characteristic_length_max = dmax

        for member_id, cyl in enumerate(cylindrical_members):
            try:
                cylinders = meshMember(
                    geom, cyl.get("heading", [0]), cyl["rA"], cyl["rB"], cyl.get("radius"),
                    member_id=member_id,
                    stations=cyl.get("stations", [0.0, 1.0]), diameters=cyl.get("diameters"), extensionA=cyl.get("extensionA", 0),  extensionB=cyl.get("extensionB", 0)
                )
                all_shapes.extend(cylinders)
            except Exception as e:
                print(f"Failed to mesh cylindrical member {member_id}: {e}")

        for rect_id, rect in enumerate(rectangular_members):
            try:
                boxes = meshRectangularMember(
                    geom, rect.get("heading", [0]), rect["rA"], rect["rB"],
                    rect["widths"], rect["heights"],
                    member_id=rect_id, stations=rect.get("stations", [0.0, 1.0]), extensionA=rect.get("extensionA", 0), extensionB=rect.get("extensionB", 0)
                )
                all_shapes.extend(boxes)
            except Exception as e:
                print(f"Failed to mesh rectangular member {rect_id}: {e}")

        if not all_shapes:
            print("No shapes were added! Check your YAML or input logic.")
            return
        
        if os.path.isdir(meshDir) is not True:
            os.makedirs(meshDir)

        try:
            combined = geom.boolean_union(all_shapes)
            geom.add_physical(combined, label="CombinedGeometry")
            mesh = geom.generate_mesh()
            stl_path = os.path.join(meshDir, "Platform.stl")
            mesh.write(stl_path)
        except Exception as e:
            print(f"Boolean union or meshing failed: {e}")
            return

    try:
        mesh_path = os.path.join(meshDir, "HullMesh.pnl")
        intermediate_path = os.path.join(meshDir, "Platform.pnl")
        if platform.system() == "Windows":

            subprocess.run([
                "meshmagick.exe",
                stl_path, "-o", intermediate_path,
                "--input-format", "stl", "--output-format", "pnl"
            ], check=True)

            subprocess.run([
                "meshmagick.exe",
                intermediate_path, "-c", "Oxy", "-o", mesh_path
            ], check=True)
        else:
            subprocess.run([
                "meshmagick",
                stl_path, "-o", intermediate_path,
                "--input-format", "stl", "--output-format", "pnl"
            ], check=True)

            subprocess.run([
                "meshmagick",
                intermediate_path, "-c", "Oxy", "-o", mesh_path
            ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Meshmagick failed: {e}")

if __name__ == "__main__":
    mesh()
