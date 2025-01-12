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

    # # cuboid annotation RAW DATA - Correct
    # pcd_filepath = r"C:\Users\Shadi\Desktop\customers\leddartech-dataloop\10.pcd"
    # annotation_rotation = [0, 0, -0.7916226654]
    # annotation_translation = [-6.903, -6.052, 0.763]
    # annotation_scale = [1.94, 4.302, 1.858]
    #
    # # pcd RAW data
    # APPLY_CORNERS_TRANSFORM = True
    # position = {
    #     "x": 5.868750585630898,
    #     "y": 5.872713278295948,
    #     "z": 0.09327199529532666
    # }
    # heading = {
    #     "w": 0.9237150445032022,
    #     "x": 0.014037729548711754,
    #     "y": 0.007406250847523422,
    #     "z": -0.38275136336262347
    # }

    # cuboid annotation DL DATA FRAME 10
    # APPLY_CORNERS_TRANSFORM = False
    # pcd_filepath = r"C:\Users\Shadi\Desktop\customers\leddartech-dataloop\10.pcd"
    # annotation_rotation = [0, 0, 0]
    # annotation_translation = [-17.106402218938634, 0.5085148017426137, 0]
    # annotation_scale = [4.1513579839416614, 1.9434064150937953, 1]
    #
    # ## DL PCD Data 10
    # position = {
    #     "x": 5.868750585630898,
    #     "y": 5.872713278295948,
    #     "z": 0.09327199529532666
    # }
    # heading = {
    #     "x": 0.014037729548711754,
    #     "y": 0.007406250847523422,
    #     "z": -0.38275136336262344,
    #     "w": 0.9237150445032022
    # }

    # cuboid annotation DL DATA FRAME 0
    APPLY_CORNERS_TRANSFORM = False
    pcd_filepath = r"C:\Users\Shadi\Desktop\customers\leddartech-dataloop\00.pcd"
    annotation_rotation = [0, 0, 0.015559934052749037]
    annotation_translation = [-14.165274009943387, 0.7167885243007961, 0]
    annotation_scale = [3.5705184026622856, 2.495895829157072, 1]
    # DL PCD Data FRAME 0
    position = {
        "x": 0,
        "y": 0,
        "z": 0
    }
    heading = {
        "w": 0.921826111774249,
        "x": 0.009768148296138231,
        "y": 0.024354524944503278,
        "z": -0.3867144425086316
    }

    # visualize annotation on pcd
    pcd = o3d.io.read_point_cloud(filename=pcd_filepath)

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


    yaw = annotation_rotation[2]
    center = annotation_translation
    dimensions = annotation_scale
    corners = calculate_cube_corners(center, dimensions)
    v3d = o3d.utility.Vector3dVector(corners)
    cloud = o3d.geometry.PointCloud(v3d)


    if APPLY_CORNERS_TRANSFORM:
        cloud.rotate(R.from_euler('z', yaw).as_matrix(), center=center)
        cloud.transform(pcd_transform_matrix)
    cuboid_points = np.asarray(cloud.points)

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