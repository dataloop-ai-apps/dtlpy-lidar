import dtlpy as dl
import json
import os
import numpy as np
import open3d as o3d
from open3d.cpu.pybind.geometry import PointCloud
from tqdm import tqdm

import dtlpylidar.utilities.transformations as transformations


def extract_dataloop_data(frames_item: dl.Item, frame_num: int):
    # Get frames.json data
    buffer = frames_item.download(save_locally=False)
    frames_json = json.load(fp=buffer)

    if len(frames_json["frames"]) <= frame_num:
        raise IndexError(f"Frame {frame_num} doesn't exists in the `frames.json` item")

    # Extract frame pcd data
    pcd_data = frames_json["frames"][frame_num]

    # Extract frame cameras data
    cameras = frames_json["cameras"]
    images = pcd_data["images"]
    camera_ids = [image["camera_id"] for image in images]
    camera_id_to_image_id_map = {image["camera_id"]: image["image_id"] for image in images}

    idx = 0
    cameras_data = dict()
    for camera in cameras:
        if camera["id"] in camera_ids:
            cameras_data[str(idx)] = {
                "sensor": camera["sensorsData"],
                "image_id": camera_id_to_image_id_map[camera["id"]]
            }
            idx += 1

    return pcd_data, cameras_data


def create_open_3d_scene_objects(frames_item: dl.Item, pcd_data: dict, cameras_data: dict):
    dataset = frames_item.dataset

    # Create PCD open3d object
    data_path = os.path.join(os.getcwd(), "data")
    os.makedirs(name=data_path, exist_ok=True)
    pcd_filepath = dataset.items.get(item_id=pcd_data["lidar"]["lidar_pcd_id"]).download(
        local_path=data_path,
        overwrite=True
    )
    pcd = o3d.io.read_point_cloud(filename=pcd_filepath)

    # Calculate the Quaternion
    lidar_quaternion = np.array([
        pcd_data["rotation"]["x"],
        pcd_data["rotation"]["y"],
        pcd_data["rotation"]["z"],
        pcd_data["rotation"]["w"]
    ])

    # Calculate the Position
    lidar_position = np.array([
        pcd_data["translation"]["x"],
        pcd_data["translation"]["y"],
        pcd_data["translation"]["z"]
    ])

    # Calculate the transform matrix
    lidar_rotation = transformations.rotation_matrix_from_quaternion(
        quaternion_x=lidar_quaternion[0],
        quaternion_y=lidar_quaternion[1],
        quaternion_z=lidar_quaternion[2],
        quaternion_w=lidar_quaternion[3]
    )
    lidar_transform_matrix = transformations.calc_transform_matrix(
        rotation=lidar_rotation,
        position=lidar_position
    )
    pcd.transform(lidar_transform_matrix)

    # Create Cameras open3d objects
    cameras = list()
    for idx, camera_data in cameras_data.items():
        image_item = dataset.items.get(item_id=camera_data["image_id"])

        ############################################
        # Create an Open3D camera using intrinsics #
        ############################################
        pinhole_camera = o3d.camera.PinholeCameraIntrinsic()

        width = image_item.width
        height = image_item.height
        fx = camera_data["sensor"]["intrinsicData"]["fx"]
        fy = camera_data["sensor"]["intrinsicData"]["fy"]
        cx = camera_data["sensor"]["intrinsicData"]["cx"]
        cy = camera_data["sensor"]["intrinsicData"]["cy"]
        pinhole_camera.set_intrinsics(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)

        ##########################################
        # Create the camera pose using extrinsic #
        ##########################################
        camera_pose = o3d.camera.PinholeCameraParameters()

        # Calculate the Quaternion
        camera_quaternion = np.array([
            camera_data["sensor"]["extrinsic"]["rotation"]["x"],
            camera_data["sensor"]["extrinsic"]["rotation"]["y"],
            camera_data["sensor"]["extrinsic"]["rotation"]["z"],
            camera_data["sensor"]["extrinsic"]["rotation"]["w"]
        ])

        # Calculate the Position
        camera_position = np.array([
            camera_data["sensor"]["extrinsic"]["position"]["x"],
            camera_data["sensor"]["extrinsic"]["position"]["y"],
            camera_data["sensor"]["extrinsic"]["position"]["z"]
        ])

        # Calculate the extrinsic matrix
        camera_rotation = transformations.rotation_matrix_from_quaternion(
            quaternion_x=camera_quaternion[0],
            quaternion_y=camera_quaternion[1],
            quaternion_z=camera_quaternion[2],
            quaternion_w=camera_quaternion[3]
        )
        extrinsic_matrix = transformations.calc_transform_matrix(
            rotation=camera_rotation,
            position=camera_position
        )
        camera_pose.extrinsic = lidar_transform_matrix @ extrinsic_matrix

        #####################################################
        # Create a line set to represent the camera frustum #
        #####################################################
        camera_line_set = o3d.geometry.LineSet().create_camera_visualization(
            intrinsic=pinhole_camera,
            extrinsic=np.identity(n=4),
            scale=100
        )
        camera_line_set.paint_uniform_color([1, 0, 0])
        camera_line_set.transform(camera_pose.extrinsic)
        cameras.append(camera_line_set)

        # Add axis to the camera
        camera_triangle = o3d.geometry.TriangleMesh().create_coordinate_frame(size=1.5)
        camera_triangle.transform(camera_pose.extrinsic)
        cameras.append(camera_triangle)

    return pcd, cameras


def create_open_3d_annotations_objects(frames_item: dl.Item, frame_num: int):
    annotations_data = list()
    annotations = frames_item.annotations.list()
    labels_map = frames_item.dataset.labels_flat_dict

    annotation: dl.Annotation
    for annotation in tqdm(annotations):
        # Get the updated annotation entity
        annotation = dl.annotations.get(annotation_id=annotation.id)
        if frame_num not in list(annotation.frames.keys()):
            continue
        else:
            frame_annotation = annotation.frames[frame_num]
            if frame_annotation is not None:
                annotation: dl.entities.FrameAnnotation = frame_annotation

        # Check if the annotation is a Cuboid
        if annotation.type == dl.AnnotationType.CUBE3D.value:
            # Get Cuboid Color
            annotation_label_data = labels_map.get(annotation.label, None)
            if annotation_label_data is None:
                hex_color = '000000'
            else:
                hex_color = annotation_label_data.color.lstrip('#')
            annotation_color = list(float(int(hex_color[i:i + 2], 16) / 255) for i in (0, 2, 4))

            # Get Cuboid Points
            annotation_definition: dl.Cube3d = annotation.annotation_definition
            cuboid_points = transformations.calc_cube_points(
                annotation_translation=annotation_definition.position,
                annotation_rotation=annotation_definition.rotation,
                annotation_scale=annotation_definition.scale,
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
            annotation_color_list = [annotation_color for _ in range(len(cuboid_lines))]
            line_set.colors = o3d.utility.Vector3dVector(annotation_color_list)

            # Add the LineSet object to the annotations_data list
            annotations_data.append(line_set)

    return annotations_data


def build_visualization(pcd: PointCloud, cameras: list, annotations_data: list,
                        dark_mode: bool, rgb_points_color: bool = False):
    # Initialize the Visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)
    vis.get_render_option().background_color = [0, 0, 0] if dark_mode else [1, 1, 1]
    if not rgb_points_color:
        pcd.paint_uniform_color([1, 1, 1] if dark_mode else [0, 0, 0])

    # Add the pcd
    vis.add_geometry(pcd)

    # Add the cameras
    for camera in cameras:
        vis.add_geometry(camera)

    for annotation in annotations_data:
        vis.add_geometry(annotation)

    # Run open3d Visualizer
    vis.run()


def visualize_in_open_3d(frames_item: dl.Item, frame_num: int, dark_mode: bool, rgb_points_color: bool = False):
    pcd_data, cameras_data = extract_dataloop_data(frames_item=frames_item, frame_num=frame_num)
    pcd, cameras = create_open_3d_scene_objects(frames_item=frames_item, pcd_data=pcd_data, cameras_data=cameras_data)
    annotations_data = create_open_3d_annotations_objects(frames_item=frames_item, frame_num=frame_num)
    build_visualization(pcd=pcd, cameras=cameras, annotations_data=annotations_data, dark_mode=dark_mode, rgb_points_color=rgb_points_color)


def main():
    frames_item_id = ""
    frame_num = 0
    dark_mode = True
    rgb_points_color = False

    frames_item = dl.items.get(item_id=frames_item_id)
    visualize_in_open_3d(frames_item=frames_item, frame_num=frame_num, dark_mode=dark_mode,
                         rgb_points_color=rgb_points_color)


if __name__ == '__main__':
    main()
