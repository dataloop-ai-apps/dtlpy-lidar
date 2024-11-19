from dtlpylidar.parser_base import extrinsic_calibrations
from dtlpylidar.parser_base import images_and_pcds, camera_calibrations, lidar_frame, lidar_scene
import dtlpylidar.utilities.transformations as transformations
import dtlpy as dl
import pandas as pd
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import os
import json
import uuid
import logging
import shutil
import pathlib

logger = logging.Logger(name="lidar_base_parser")


class LidarBaseParser(dl.BaseServiceRunner):
    # TODO: Override this method in the derived class if needed
    @staticmethod
    def download_data(dataset: dl.Dataset, remote_path: str, download_path) -> tuple:
        """
        Download the required data for the parser
        :param dataset: Input dataset
        :param remote_path: Path to the remote folder where the Lidar data is uploaded
        :param download_path: Path to the downloaded data
        :return:
        """
        # Download items dataloop annotation JSONs
        filters = dl.Filters(field="metadata.system.mimetype", values="*pcd*", method=dl.FiltersMethod.OR)
        filters.add(field="metadata.system.mimetype", values="*image*", method=dl.FiltersMethod.OR)
        dataset.download_annotations(local_path=download_path, filters=filters)

        # Download required binaries (Calibrations Data)
        filters = dl.Filters(field="metadata.system.mimetype", values="*json*")
        dataset.items.download(local_path=download_path, filters=filters)

        # Download required binaries (Annotations Data)
        filters = dl.Filters(field="metadata.system.mimetype", values="*csv*")
        dataset.items.download(local_path=download_path, filters=filters)

        items_path = os.path.join(download_path, "items", remote_path)
        json_path = os.path.join(download_path, "json", remote_path)
        return items_path, json_path

    # TODO: Override this method in the derived class if needed
    @staticmethod
    def parse_lidar_data(items_path, json_path) -> dict:
        lidar_data = dict()

        lidar_json_path = os.path.join(json_path, "lidar")
        lidar_items_path = os.path.join(items_path, "lidar")

        # Open the poses and timestamps JSONs
        poses_json = os.path.join(lidar_items_path, "poses.json")
        with open(poses_json, 'r') as f:
            poses_json_data: list = json.load(f)

        timestamps_json = os.path.join(lidar_items_path, "timestamps.json")
        with open(timestamps_json, 'r') as f:
            timestamps_json_data: list = json.load(f)

        # Get all the lidar JSONs sorted by frame number
        lidar_jsons = pathlib.Path(lidar_json_path).rglob('*.json')
        lidar_jsons = sorted(lidar_jsons, key=lambda x: int(x.stem))

        for lidar_frame_idx, lidar_json in enumerate(lidar_jsons):
            with open(lidar_json, 'r') as f:
                lidar_json_data = json.load(f)

            # lidar_data = {
            #     "path": pathlib.Path(lidar_json).absolute(),
            #     "timestamp": timestamps_json_data[lidar_frame_idx],
            #     "position": poses_json_data[lidar_frame_idx]["position"],
            #     "heading": poses_json_data[lidar_frame_idx]["heading"],
            #     "cameras": dict()
            # }

            ground_map_id = lidar_json_data.get("metadata", dict()).get("user", dict()).get(
                "lidar_ground_detection", dict()).get("groundMapId", None)
            lidar_translation = extrinsic_calibrations.Translation(
                x=poses_json_data[lidar_frame_idx].get("position", dict()).get("x", 0),
                y=poses_json_data[lidar_frame_idx].get("position", dict()).get("y", 0),
                z=poses_json_data[lidar_frame_idx].get("position", dict()).get("z", 0),
            )
            lidar_rotation = extrinsic_calibrations.QuaternionRotation(
                x=poses_json_data[lidar_frame_idx].get("heading", dict()).get("x", 0),
                y=poses_json_data[lidar_frame_idx].get("heading", dict()).get("y", 0),
                z=poses_json_data[lidar_frame_idx].get("heading", dict()).get("z", 0),
                w=poses_json_data[lidar_frame_idx].get("heading", dict()).get("w", 1)
            )
            lidar_timestamp = str(timestamps_json_data[lidar_frame_idx])

            lidar_pcd_data = images_and_pcds.LidarPcdData(
                item_id=lidar_json_data.get("id"),
                ground_id=ground_map_id,
                remote_path=lidar_json_data.get("filename"),
                extrinsic=extrinsic_calibrations.Extrinsic(
                    rotation=lidar_rotation,
                    translation=lidar_translation
                ),
                timestamp=lidar_timestamp
            )
            lidar_data[lidar_frame_idx] = lidar_pcd_data

        return lidar_data

    # TODO: Override this method in the derived class if needed
    @staticmethod
    def parse_cameras_data(items_path, json_path) -> dict:
        cameras_data = dict()

        camera_json_path = os.path.join(json_path, "camera")
        camera_items_path = os.path.join(items_path, "camera")

        # camera_list = [
        #     'front_camera',
        #     'front_left_camera',
        #     'left_camera',
        #     'back_camera',
        #     'right_camera',
        #     'front_right_camera'
        # ]

        camera_folders_list = sorted(os.listdir(camera_json_path))
        for camera_folder_idx, camera_folder in enumerate(camera_folders_list):
            cameras_data[camera_folder] = dict()

            camera_folder_json_path = os.path.join(camera_json_path, camera_folder)
            camera_folder_items_path = os.path.join(camera_items_path, camera_folder)

            # Open the poses, intrinsics and timestamps JSONs
            intrinsics_json = os.path.join(camera_folder_items_path, "intrinsics.json")
            with open(intrinsics_json, 'r') as f:
                intrinsics_json_data: dict = json.load(f)

            poses_json = os.path.join(camera_folder_items_path, "poses.json")
            with open(poses_json, 'r') as f:
                poses_json_data: list = json.load(f)

            timestamps_json = os.path.join(camera_folder_items_path, "timestamps.json")
            with open(timestamps_json, 'r') as f:
                timestamps_json_data: list = json.load(f)

            # Get all the camera JSONs sorted by frame number
            camera_jsons = pathlib.Path(camera_folder_json_path).rglob('*.json')
            camera_jsons = sorted(camera_jsons, key=lambda x: int(x.stem))

            for camera_frame_idx, camera_json in enumerate(camera_jsons):
                with open(camera_json, 'r') as f:
                    camera_json_data = json.load(f)

                camera_id = f"{camera_folder}_frame_{camera_frame_idx}"
                camera_intrinsic = camera_calibrations.Intrinsic(
                    fx=intrinsics_json_data.get("fx", 0),
                    fy=intrinsics_json_data.get("fy", 0),
                    cx=intrinsics_json_data.get("cx", 0),
                    cy=intrinsics_json_data.get("cy", 0)
                )
                camera_rotation = extrinsic_calibrations.QuaternionRotation(
                    x=poses_json_data[camera_frame_idx].get("heading", dict()).get("x", 0),
                    y=poses_json_data[camera_frame_idx].get("heading", dict()).get("y", 0),
                    z=poses_json_data[camera_frame_idx].get("heading", dict()).get("z", 0),
                    w=poses_json_data[camera_frame_idx].get("heading", dict()).get("w", 1)
                )
                camera_translation = extrinsic_calibrations.Translation(
                    x=poses_json_data[camera_frame_idx].get("position", dict()).get("x", 0),
                    y=poses_json_data[camera_frame_idx].get("position", dict()).get("y", 0),
                    z=poses_json_data[camera_frame_idx].get("position", dict()).get("z", 0)
                )
                camera_distortion = camera_calibrations.Distortion(
                    k1=0,
                    k2=0,
                    k3=0,
                    p1=0,
                    p2=0
                )
                camera_timestamp = str(timestamps_json_data[camera_frame_idx])

                lidar_camera_data = camera_calibrations.LidarCameraData(
                    intrinsic=camera_intrinsic,
                    extrinsic=extrinsic_calibrations.Extrinsic(
                        rotation=camera_rotation,
                        translation=camera_translation
                    ),
                    channel=camera_json_data.get("filename"),
                    distortion=camera_distortion,
                    cam_id=camera_id,
                )

                lidar_image_data = images_and_pcds.LidarImageData(
                    item_id=camera_json_data.get("id"),
                    lidar_camera=lidar_camera_data,
                    remote_path=camera_json_data.get("filename"),
                    timestamp=camera_timestamp
                )

                cameras_data[camera_folder][camera_frame_idx] = {
                    "lidar_camera": lidar_camera_data,
                    "lidar_image": lidar_image_data
                }

        return cameras_data

    @staticmethod
    def _calculate_cube_corners(center, dimensions):
        half_dimensions = np.array(dimensions) / 2
        corner_offsets = [
            [-1, -1, -1], [+1, -1, -1], [-1, +1, -1], [+1, +1, -1],
            [-1, -1, +1], [+1, -1, +1], [-1, +1, +1], [+1, +1, +1]
        ]
        corner_offsets = np.array(corner_offsets)
        corners = center + corner_offsets * half_dimensions
        return corners

    @staticmethod
    def _fix_box_directions(corners, transformation_matrix, yaw):
        v3d = o3d.utility.Vector3dVector(corners)
        cloud = o3d.geometry.PointCloud(v3d)
        cloud.transform(transformation_matrix)
        box_rotation = R.from_euler('xyz', [0, 0, float(yaw)]).as_matrix()
        new_box_rotation = list(
            R.from_matrix(np.dot(transformation_matrix[:3, :3], box_rotation)).as_euler('xyz'))
        new_box_translation = cloud.get_center() - transformation_matrix[3, :3]
        return new_box_translation, new_box_rotation

    def _calculate_cube_transform(self, row_data, frame_pcd_translation, frame_pcd_rotation):
        ann_position = np.array([row_data["position.x"], row_data["position.y"], row_data["position.z"]])
        ann_rotation = np.array([0, 0, row_data["yaw"]])
        ann_scale = np.array([row_data["dimensions.x"], row_data["dimensions.y"], row_data["dimensions.z"]])
        bbox_corners = self._calculate_cube_corners(center=ann_position, dimensions=ann_scale)

        transformation_matrix = transformations.calc_transform_matrix(
            rotation=R.from_quat(frame_pcd_rotation).as_matrix(),
            position=np.array(frame_pcd_translation)
        )
        new_ann_translation, new_ann_rotation = self._fix_box_directions(
            corners=bbox_corners,
            transformation_matrix=transformation_matrix,
            yaw=ann_rotation[-1],
        )
        return new_ann_translation, new_ann_rotation

    # TODO: Override this method in the derived class if needed
    def parse_annotations(self, frames_item: dl.Item, items_path, json_path):
        annotations_json_path = os.path.join(json_path, "annotations")
        annotations_items_path = os.path.join(items_path, "annotations")

        builder = frames_item.annotations.builder()
        frames_json_data = json.loads(s=frames_item.download(save_locally=False).getvalue())

        next_object_id = 0
        uid_to_object_id_map = dict()

        ################################
        # Parse the cuboid annotations #
        ################################
        cuboids_items_path = os.path.join(annotations_items_path, "cuboids")
        cuboids_csvs = pathlib.Path(cuboids_items_path).rglob('*.csv')
        cuboids_csvs = sorted(cuboids_csvs, key=lambda x: int(x.stem))

        for csv_frame_idx, cuboids_csv in enumerate(cuboids_csvs):
            frame_pcd_translation = frames_json_data["frames"][csv_frame_idx]["translation"]
            frame_pcd_translation = np.array(
                [frame_pcd_translation["x"], frame_pcd_translation["y"], frame_pcd_translation["z"]]
            )
            frame_pcd_rotation = frames_json_data["frames"][csv_frame_idx]["rotation"]
            frame_pcd_rotation = np.array(
                [frame_pcd_rotation["x"], frame_pcd_rotation["y"], frame_pcd_rotation["z"], frame_pcd_rotation["w"]]
            )

            cuboids_csv_data = pd.read_csv(filepath_or_buffer=cuboids_csv)
            for _, row_data in cuboids_csv_data.iterrows():
                object_id = uid_to_object_id_map.get(row_data["uuid"], None)
                if object_id is None:
                    object_id = next_object_id
                    uid_to_object_id_map[row_data["uuid"]] = object_id
                    next_object_id += 1

                ann_label = row_data["label"]
                ann_scale = np.array([row_data["dimensions.x"], row_data["dimensions.y"], row_data["dimensions.z"]])
                ann_position, ann_rotation = self._calculate_cube_transform(
                    row_data=row_data,
                    frame_pcd_translation=frame_pcd_translation,
                    frame_pcd_rotation=frame_pcd_rotation
                )

                annotation_definition = dl.Cube3d(
                    label=ann_label,
                    position=ann_position,
                    scale=ann_scale,
                    rotation=ann_rotation
                )
                builder.add(
                    annotation_definition=annotation_definition,
                    object_id=object_id,
                    frame_num=csv_frame_idx
                )

        ################################
        # Parse the semseg annotations #
        ################################

        semseg_items_path = os.path.join(annotations_items_path, "semseg")
        semseg_csvs = pathlib.Path(semseg_items_path).rglob('*.csv')
        semseg_csvs = sorted(semseg_csvs, key=lambda x: int(x.stem))

        for csv_frame_idx, semseg_csv in enumerate(semseg_csvs):
            # TODO: find class to label mapping
            pass

        builder.upload()

    # TODO: Add to docs to first convert the PLY to PCD
    @staticmethod
    def build_lidar_scene(lidar_data: dict, cameras_data: dict):
        """
        Convert the calibration data to a LidarScene object
        :return: buffer: BytesIO buffer with the LidarScene data to be uploaded to the dataloop platform as JSON
        """
        scene = lidar_scene.LidarScene()
        frames_number = len(lidar_data)
        for frame_number in range(frames_number):
            logger.info(f"Processing PCD data [Frame: {frame_number}]")
            frame_lidar_pcd_data = lidar_data[frame_number]
            lidar_frame_images = list()

            for camera_idx, (camera_folder, camera_data) in enumerate(cameras_data.items()):
                logger.info(f"Processing Camera data [Frame: {frame_number}, Camera Index: {camera_idx}]")
                frame_lidar_camera_data = camera_data.get(frame_number, dict()).get("lidar_camera", None)
                frame_lidar_image_data = camera_data.get(frame_number, dict()).get("lidar_image", None)

                if frame_lidar_camera_data is None or frame_lidar_image_data is None:
                    continue

                scene.add_camera(frame_lidar_camera_data)
                lidar_frame_images.append(frame_lidar_image_data)

            frame_item = lidar_frame.LidarSceneFrame(
                lidar_frame_pcd=frame_lidar_pcd_data,
                lidar_frame_images=lidar_frame_images
            )
            scene.add_frame(frame_item)

        scene_data = scene.to_json()
        return scene_data

    def run(self, dataset: dl.Dataset, remote_path: str = "/"):
        """
        Run the parser
        :param dataset: Input dataset
        :param remote_path: Path to the remote folder where the Lidar data is uploaded
        :return: frames_item: dl.Item entity of the uploaded frames.json
        """
        if remote_path.startswith("/"):
            remote_path = remote_path[1:]

        if remote_path.endswith("/"):
            remote_path = remote_path[:-1]

        base_path = f"{dataset.name}_{str(uuid.uuid4())}"
        download_path = os.path.join(os.getcwd(), base_path)
        try:
            items_path, json_path = self.download_data(
                dataset=dataset,
                remote_path=remote_path,
                download_path=download_path
            )

            # lidar_data = self.parse_lidar_data(items_path=items_path, json_path=json_path)
            # cameras_data = self.parse_cameras_data(items_path=items_path, json_path=json_path)
            # scene_data = self.build_lidar_scene(lidar_data=lidar_data, cameras_data=cameras_data)

            # frames_item = dataset.items.upload(
            #     remote_name="frames.json",
            #     remote_path=f"/{remote_path}",
            #     local_path=json.dumps(scene_data).encode(),
            #     overwrite=True,
            #     item_metadata={
            #         "system": {
            #             "shebang": {
            #                 "dltype": "PCDFrames"
            #             }
            #         },
            #         "fps": 1
            #     }
            # )
            frames_item = dl.items.get(item_id="673a1c4d94a8e6395e093ef0")
            self.parse_annotations(frames_item=frames_item, items_path=items_path, json_path=json_path)
        finally:
            shutil.rmtree(path=base_path, ignore_errors=True)

        return frames_item


def test_parser():
    dataset = dl.datasets.get(dataset_id="673615818ab4c9a0b0be683e")
    parser = LidarBaseParser()
    print(parser.run(dataset=dataset))


if __name__ == '__main__':
    test_parser()
