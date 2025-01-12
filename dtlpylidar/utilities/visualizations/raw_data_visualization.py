import open3d as o3d
import numpy as np
from dtlpylidar.utilities import transformations
from scipy.spatial.transform import Rotation as R


def calculate_cube_corners(center, dimensions):
    # Half dimensions for easier calculations
    half_dimensions = np.array(dimensions) / 2

    # Create a list of corner offsets relative to the center
    corner_offsets = [
        [-1, -1, -1], [+1, -1, -1], [-1, +1, -1], [+1, +1, -1],
        [-1, -1, +1], [+1, -1, +1], [-1, +1, +1], [+1, +1, +1]
    ]

    # Convert corner offsets to numpy array
    corner_offsets = np.array(corner_offsets)

    # Calculate corner positions by adding offsets to center and multiplying by half dimensions
    corners = center + corner_offsets * half_dimensions

    return corners


# 8de8c466-c0b7-48c1-8218-693636888bf6	Car	-0.7916226654	False	-1	-6.903	-6.052	0.763	1.94	4.302	1.858	Moving	-0	-1


if __name__ == '__main__':
    # Flags
    APPLY_PCD_TRANSFORM = False
    APPLY_CORNERS_TRANSFORM = False
    APPLY_ANNOTATION_TRANSFORM = True

    # cuboid annotation
    annotation_rotation = [0, 0, -0.7916226654]
    annotation_translation = [-6.903, -6.052, 0.763]
    annotation_scale = [1.94, 4.302, 1.858]

    # pcd data
    position = {
        "x": 5.868750585630898,
        "y": 5.872713278295948,
        "z": 0.09327199529532666
    }
    heading = {
        "w": 0.9237150445032022,
        "x": 0.014037729548711754,
        "y": 0.007406250847523422,
        "z": -0.38275136336262347
    }

    # PCD DATA
    pcd_filepath = r"C:\Users\Ofir\PycharmProjects\dtlpy-lidar\dtlpylidar\utilities\visualizations\10.pcd"
    pcd = o3d.io.read_point_cloud(filename=pcd_filepath)

    # Apply PCD Transform
    if APPLY_PCD_TRANSFORM:
        rotation = transformations.rotation_matrix_from_quaternion(
            quaternion_x=heading["x"],
            quaternion_y=heading["y"],
            quaternion_z=heading["z"],
            quaternion_w=heading["w"]
        )
        position = np.array([position["x"], position["y"], position["z"]])
        pcd_transform_matrix = transformations.calc_transform_matrix(
            rotation=rotation,
            position=position
        )
        pcd.transform(pcd_transform_matrix)

    # Apply Corners Transform
    if APPLY_CORNERS_TRANSFORM:
        yaw = annotation_rotation[2]
        center = annotation_translation
        dimensions = annotation_scale
        corners = calculate_cube_corners(center, dimensions)
        v3d = o3d.utility.Vector3dVector(corners)
        cloud = o3d.geometry.PointCloud(v3d)
        cloud.transform(pcd_transform_matrix)
        # TODO: Use dtlpy lidar SDK
        box_rotation = R.from_euler('xyz', [0, 0, float(yaw)]).as_matrix()
        new_box_rotation = list(
            R.from_matrix(np.dot(pcd_transform_matrix[:3, :3], box_rotation)).as_euler('xyz')
        )
        annotation_rotation = new_box_rotation

    scale_x = annotation_scale[0]
    scale_y = annotation_scale[1]
    scale_z = annotation_scale[2]

    position_x = annotation_translation[0]
    position_y = annotation_translation[1]
    position_z = annotation_translation[2]

    cuboid_points = np.asarray([
        [position_x + scale_x / 2, position_y + scale_y / 2, position_z + scale_z / 2],
        [position_x + scale_x / 2, position_y + scale_y / 2, position_z - scale_z / 2],
        [position_x + scale_x / 2, position_y - scale_y / 2, position_z + scale_z / 2],
        [position_x + scale_x / 2, position_y - scale_y / 2, position_z - scale_z / 2],
        [position_x - scale_x / 2, position_y + scale_y / 2, position_z + scale_z / 2],
        [position_x - scale_x / 2, position_y + scale_y / 2, position_z - scale_z / 2],
        [position_x - scale_x / 2, position_y - scale_y / 2, position_z + scale_z / 2],
        [position_x - scale_x / 2, position_y - scale_y / 2, position_z - scale_z / 2],
    ])


    if APPLY_ANNOTATION_TRANSFORM is True:
        if annotation_rotation is None:
            raise Exception("Rotation must be provided")
        rotation_x = annotation_rotation[0]
        rotation_y = annotation_rotation[1]
        rotation_z = annotation_rotation[2]
        rotation_matrix = transformations.rotation_matrix_from_euler(
            rotation_x=rotation_x,
            rotation_y=rotation_y,
            rotation_z=rotation_z
        )
        translation_matrix = transformations.calc_translation_matrix(
            position_x=position_x,
            position_y=position_y,
            position_z=position_z
        )
        cuboid_points = transformations.rotate_annotation_cube3d(
            annotation_corners=cuboid_points,
            rotation_matrix=rotation_matrix,
            translation_matrix=translation_matrix
        )

    # Get Cuboid Lines
    cuboid_lines = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]

    # Create LineSet object of the Cuboid
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(cuboid_points),
                                    lines=o3d.utility.Vector2iVector(cuboid_lines))
    annotation_color = [1, 0, 0]
    annotation_color_list = [annotation_color for _ in range(len(cuboid_lines))]
    line_set.colors = o3d.utility.Vector3dVector(annotation_color_list)

    # Initialize the Visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)
    vis.get_render_option().background_color = [0, 0, 0]
    pcd.paint_uniform_color([1, 1, 1])

    # Add the pcd
    vis.add_geometry(pcd)

    vis.add_geometry(line_set)

    # Run open3d Visualizer
    vis.run()
