import dtlpy as dl
import json
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation
import math


class FixTransformation:
    @staticmethod
    def rotate_system(theta_x=None, theta_y=None, theta_z=None, radians: bool = True):
        if radians is False:
            theta_x = math.radians(theta_x) if theta_x else None
            theta_y = math.radians(theta_y) if theta_y else None
            theta_z = math.radians(theta_z) if theta_z else None

        rotation = np.identity(n=4)
        if theta_x is not None:
            rotation_x = np.array([
                [1, 0, 0, 0],
                [0, math.cos(theta_x), -math.sin(theta_x), 0],
                [0, math.sin(theta_x), math.cos(theta_x), 0],
                [0, 0, 0, 1]
            ])
            rotation = rotation @ rotation_x
        if theta_y is not None:
            rotation_y = np.array([
                [math.cos(theta_y), 0, math.sin(theta_y), 0],
                [0, 1, 0, 0],
                [-math.sin(theta_y), 0, math.cos(theta_y), 0],
                [0, 0, 0, 1]
            ])
            rotation = rotation @ rotation_y
        if theta_z is not None:
            rotation_z = np.array([
                [math.cos(theta_z), -math.sin(theta_z), 0, 0],
                [math.sin(theta_z), math.cos(theta_z), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            rotation = rotation @ rotation_z
        rotation[np.abs(rotation) < 1e-5] = 0
        return rotation

    @staticmethod
    def translate_system(x=None, y=None, z=None):
        translation = np.identity(n=4)
        if x is not None:
            translation[0, 3] = x
        if y is not None:
            translation[1, 3] = y
        if z is not None:
            translation[2, 3] = z
        return translation


def extract_dataloop_data(frames_item: dl.Item, frame_num: int):
    # Get frames.json data
    buffer = frames_item.download(save_locally=False)
    frames_json = json.load(fp=buffer)
    frame_data = frames_json["frames"][frame_num]
    cameras = frames_json["cameras"]

    # Extract frame pcd data
    pcd_data = {
        "translation": frame_data["translation"],
        "rotation": frame_data["rotation"],
        "pcd_id": frame_data["lidar"]["lidar_pcd_id"]
    }

    # Extract frame cameras data
    images = frame_data["images"]
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


def create_open_3d_objects(frames_item: dl.Item, pcd_data: dict, cameras_data: dict):
    dataset = frames_item.dataset

    # Create PCD open3d object
    pcd_filepath = dataset.items.get(item_id=pcd_data["pcd_id"]).download(local_path=".")
    pcd = o3d.io.read_point_cloud(filename=pcd_filepath)

    # Calculate the Rotation
    quaternion = np.array([
        pcd_data["rotation"]["x"],
        pcd_data["rotation"]["y"],
        pcd_data["rotation"]["z"],
        pcd_data["rotation"]["w"]
    ])
    rotation = Rotation.from_quat(quaternion).as_matrix()

    # Calculate the Position
    position = np.array([
        pcd_data["translation"]["x"],
        pcd_data["translation"]["y"],
        pcd_data["translation"]["z"]
    ])

    # Calculate the extrinsic matrix
    lidar_transform_matrix = np.identity(n=4)
    lidar_transform_matrix[0: 3, 0: 3] = rotation
    lidar_transform_matrix[0: 3, 3] = position
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

        # Calculate the Rotation
        quaternion = np.array([
            camera_data["sensor"]["extrinsic"]["rotation"]["x"],
            camera_data["sensor"]["extrinsic"]["rotation"]["y"],
            camera_data["sensor"]["extrinsic"]["rotation"]["z"],
            camera_data["sensor"]["extrinsic"]["rotation"]["w"]
        ])
        rotation = Rotation.from_quat(quaternion).as_matrix()

        # Calculate the Position
        position = np.array([
            camera_data["sensor"]["extrinsic"]["position"]["x"],
            camera_data["sensor"]["extrinsic"]["position"]["y"],
            camera_data["sensor"]["extrinsic"]["position"]["z"]
        ])

        # Calculate the extrinsic matrix
        extrinsic_matrix = np.identity(n=4)
        extrinsic_matrix[0: 3, 0: 3] = rotation
        extrinsic_matrix[0: 3, 3] = position
        camera_pose.extrinsic = extrinsic_matrix

        #####################################################
        # Create a line set to represent the camera frustum #
        #####################################################
        # TODO: find how to use create_camera_visualization correctly
        # camera_line_set = o3d.geometry.LineSet.create_camera_visualization(
        #     view_width_px=width,
        #     view_height_px=height,
        #     intrinsic=pinhole_camera.intrinsic_matrix,
        #     extrinsic=camera_pose.extrinsic,
        # )
        # camera_line_set.paint_uniform_color([1, 0, 0])
        # cameras.append(camera_line_set)

        camera_triangle = o3d.geometry.TriangleMesh().create_coordinate_frame(size=1.5)
        camera_triangle.transform(extrinsic_matrix)
        cameras.append(camera_triangle)

    return pcd, cameras


def build_visualization(pcd: o3d.cpu.pybind.geometry.PointCloud, cameras: list, dark_mode: bool):
    # Initialize the Visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)
    vis.get_render_option().background_color = [0, 0, 0] if dark_mode else [1, 1, 1]

    # Add the pcd
    pcd.paint_uniform_color([1, 1, 1] if dark_mode else [0, 0, 0])
    vis.add_geometry(pcd)

    # Add the cameras
    for camera in cameras:
        vis.add_geometry(camera)

    # Run open3d Visualizer
    vis.run()


def visualize_in_open_3d(frames_item: dl.Item, frame_num: int, dark_mode: bool):
    pcd_data, cameras_data = extract_dataloop_data(frames_item=frames_item, frame_num=frame_num)
    pcd, cameras = create_open_3d_objects(frames_item=frames_item, pcd_data=pcd_data, cameras_data=cameras_data)
    build_visualization(pcd=pcd, cameras=cameras, dark_mode=dark_mode)


def main():
    dl.setenv('prod')
    item_id = "657acd78d92fde479517c7d8"
    frame_num = 0
    dark_mode = True

    frames_item = dl.items.get(item_id=item_id)
    visualize_in_open_3d(frames_item=frames_item, frame_num=frame_num, dark_mode=dark_mode)


if __name__ == '__main__':
    main()
