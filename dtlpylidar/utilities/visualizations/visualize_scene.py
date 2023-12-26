import dtlpy as dl
import json
import open3d as o3d
import numpy as np
from open3d.cpu.pybind.geometry import PointCloud
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


def create_open_3d_objects(frames_item: dl.Item, pcd_data: dict, cameras_data: dict):
    dataset = frames_item.dataset

    # Create PCD open3d object
    pcd_filepath = dataset.items.get(item_id=pcd_data["lidar"]["lidar_pcd_id"]).download(local_path=".")
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


def build_visualization(pcd: PointCloud, cameras: list, dark_mode: bool):
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
    frames_item_id = "<frames-item-id>"
    frame_num = 0
    dark_mode = True

    frames_item = dl.items.get(item_id=frames_item_id)
    visualize_in_open_3d(frames_item=frames_item, frame_num=frame_num, dark_mode=dark_mode)


if __name__ == '__main__':
    main()
