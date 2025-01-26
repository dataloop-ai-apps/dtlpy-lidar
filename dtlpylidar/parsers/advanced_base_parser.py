import dtlpy as dl
import os
import json
import uuid
import logging
import shutil
import pathlib
import numpy as np

from dtlpylidar.parser_base import (extrinsic_calibrations, images_and_pcds, camera_calibrations, lidar_frame,
                                    lidar_scene)
from dtlpylidar.utilities import transformations

logger = logging.Logger(name="advanced_base_parser")


class AdvancedBaseParser(dl.BaseServiceRunner):
    # TODO: Override this method in the derived class if needed
    @staticmethod
    def download_data(dataset: dl.Dataset, remote_path: str, download_path: str) -> tuple:
        """
        Download the required data for the parser
        :param dataset: Input dataset
        :param remote_path: Path to the remote folder where the LiDAR data is uploaded
        :param download_path: Path to the downloaded data
        :return: (items_path, json_path) Paths to the downloaded items and annotations JSON files directories
        """
        # Download items dataloop annotation JSONs
        # (PCD and Image annotation JSONs contains the Dataloop platform references (Like: ID) to the remote files)
        filters = dl.Filters(field="metadata.system.mimetype", values="*pcd*", method=dl.FiltersMethod.OR)
        filters.add(field="metadata.system.mimetype", values="*image*", method=dl.FiltersMethod.OR)
        dataset.download_annotations(local_path=download_path, filters=filters)

        # Download required binaries (Calibrations Data)
        # PandaSet Calibration Data is saved in JSON files (Like: poses.json, intrinsics.json, timestamps.json)
        filters = dl.Filters(field="metadata.system.mimetype", values="*json*")
        dataset.items.download(local_path=download_path, filters=filters)

        items_path = os.path.join(download_path, "items", remote_path)
        json_path = os.path.join(download_path, "json", remote_path)
        return items_path, json_path

    # TODO: Override this method in the derived class if needed
    @staticmethod
    def parse_lidar_data(items_path: str, json_path: str) -> dict:
        """
        Parse the LiDAR Calibration data to build all the scene LidarPcdData objects
        :param items_path: Paths to the downloaded items directory
        :param json_path: Paths to the downloaded annotations JSON files directory
        :return: lidar_data: Dictionary containing mapping of frame number to LidarPcdData object
        """
        lidar_data = dict()

        lidar_json_path = os.path.join(json_path, "lidar")
        lidar_items_path = os.path.join(items_path, "lidar")

        # NOTE: Not supported
        # # Opening the poses.json file to get the Extrinsic (Translation and Rotation) of the Lidar Scene per frame
        # poses_json = os.path.join(lidar_items_path, "poses.json")
        # with open(poses_json, 'r') as f:
        #     poses_json_data: list = json.load(f)

        # Opening the poses.json file to get the Timestamps of the Lidar Scene per frame
        timestamps_json = os.path.join(lidar_items_path, "timestamps.json")
        with open(timestamps_json, 'r') as f:
            timestamps_json_data: list = json.load(f)

        # Get all the lidar JSONs sorted by frame number
        lidar_jsons = pathlib.Path(lidar_json_path).rglob('*.json')
        lidar_jsons = sorted(lidar_jsons, key=lambda x: int(x.stem))

        for lidar_frame_idx, lidar_json in enumerate(lidar_jsons):
            with open(lidar_json, 'r') as f:
                lidar_json_data = json.load(f)

            ground_map_id = lidar_json_data.get("metadata", dict()).get("user", dict()).get(
                "lidar_ground_detection", dict()).get("groundMapId", None)

            # NOTE: Not supported
            # lidar_translation = extrinsic_calibrations.Translation(
            #     x=poses_json_data[lidar_frame_idx].get("position", dict()).get("x", 0),
            #     y=poses_json_data[lidar_frame_idx].get("position", dict()).get("y", 0),
            #     z=poses_json_data[lidar_frame_idx].get("position", dict()).get("z", 0),
            # )
            # lidar_rotation = extrinsic_calibrations.QuaternionRotation(
            #     x=poses_json_data[lidar_frame_idx].get("heading", dict()).get("x", 0),
            #     y=poses_json_data[lidar_frame_idx].get("heading", dict()).get("y", 0),
            #     z=poses_json_data[lidar_frame_idx].get("heading", dict()).get("z", 0),
            #     w=poses_json_data[lidar_frame_idx].get("heading", dict()).get("w", 1)
            # )

            lidar_translation = extrinsic_calibrations.Translation(x=0, y=0, z=0)
            lidar_rotation = extrinsic_calibrations.QuaternionRotation(x=0, y=0, z=0, w=1)
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
    def parse_cameras_data(items_path: str, json_path: str) -> dict:
        """
        Parse the Cameras Calibration data to build all the scene LidarCameraData and LidarImageData objects
        :param items_path: Paths to the downloaded items directory
        :param json_path: Paths to the downloaded annotations JSON files directory
        :return: lidar_data: Dictionary containing mapping of camera and frame number
        to LidarCameraData and LidarImageData objects
        """
        lidar_transform_list = list()

        lidar_items_path = os.path.join(items_path, "lidar")
        lidar_json_path = os.path.join(json_path, "lidar")

        poses_json = os.path.join(lidar_items_path, "poses.json")
        with open(poses_json, 'r') as f:
            poses_json_data: list = json.load(f)

        # Get count of all the lidar JSONs
        lidar_jsons = pathlib.Path(lidar_json_path).rglob('*.json')
        lidar_jsons_count = len(list(lidar_jsons))

        for idx in range(lidar_jsons_count):
            lidar_position_idx = [
                poses_json_data[idx].get("position", dict()).get("x", 0),
                poses_json_data[idx].get("position", dict()).get("y", 0),
                poses_json_data[idx].get("position", dict()).get("z", 0)
            ]
            lidar_heading_idx = [
                poses_json_data[idx].get("heading", dict()).get("x", 0),
                poses_json_data[idx].get("heading", dict()).get("y", 0),
                poses_json_data[idx].get("heading", dict()).get("z", 0),
                poses_json_data[idx].get("heading", dict()).get("w", 0)
            ]
            lidar_rotation_idx = transformations.rotation_matrix_from_quaternion(*lidar_heading_idx)
            lidar_transform_idx = transformations.calc_transform_matrix(
                rotation=lidar_rotation_idx,
                position=lidar_position_idx
            )
            lidar_transform_list.append(lidar_transform_idx)

        cameras_data = dict()

        camera_json_path = os.path.join(json_path, "camera")
        camera_items_path = os.path.join(items_path, "camera")

        # Get the list of all the available camera folders, and building the cameras data objects per camera per frame
        camera_folders_list = sorted(os.listdir(camera_json_path))
        for camera_folder_idx, camera_folder in enumerate(camera_folders_list):
            cameras_data[camera_folder] = dict()

            camera_folder_json_path = os.path.join(camera_json_path, camera_folder)
            camera_folder_items_path = os.path.join(camera_items_path, camera_folder)

            # Opening the intrinsics.json file to get the Intrinsics (fx, fy, cx, cy) of the Current Camera per frame
            intrinsics_json = os.path.join(camera_folder_items_path, "intrinsics.json")
            with open(intrinsics_json, 'r') as f:
                intrinsics_json_data: dict = json.load(f)

            # Opening the poses.json file to get the Extrinsic (Translation and Rotation) of the Current Camera per frame
            poses_json = os.path.join(camera_folder_items_path, "poses.json")
            with open(poses_json, 'r') as f:
                poses_json_data: list = json.load(f)

            # Opening the poses.json file to get the Timestamps of the Current Camera per frame
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

                # NOTE: Not supported
                # camera_translation = extrinsic_calibrations.Translation(
                #     x=poses_json_data[camera_frame_idx].get("position", dict()).get("x", 0),
                #     y=poses_json_data[camera_frame_idx].get("position", dict()).get("y", 0),
                #     z=poses_json_data[camera_frame_idx].get("position", dict()).get("z", 0)
                # )
                # camera_rotation = extrinsic_calibrations.QuaternionRotation(
                #     x=poses_json_data[camera_frame_idx].get("heading", dict()).get("x", 0),
                #     y=poses_json_data[camera_frame_idx].get("heading", dict()).get("y", 0),
                #     z=poses_json_data[camera_frame_idx].get("heading", dict()).get("z", 0),
                #     w=poses_json_data[camera_frame_idx].get("heading", dict()).get("w", 1)
                # )

                # Get lidar transform for the current camera frame
                lidar_transform_idx = lidar_transform_list[camera_frame_idx]

                # Get camera transform for the current camera frame
                camera_position_idx = [
                    poses_json_data[camera_frame_idx].get("position", dict()).get("x", 0),
                    poses_json_data[camera_frame_idx].get("position", dict()).get("y", 0),
                    poses_json_data[camera_frame_idx].get("position", dict()).get("z", 0)
                ]
                camera_heading_idx = [
                    poses_json_data[camera_frame_idx].get("heading", dict()).get("x", 0),
                    poses_json_data[camera_frame_idx].get("heading", dict()).get("y", 0),
                    poses_json_data[camera_frame_idx].get("heading", dict()).get("z", 0),
                    poses_json_data[camera_frame_idx].get("heading", dict()).get("w", 1)
                ]

                # Calculate updated camera transform
                camera_rotation_idx = transformations.rotation_matrix_from_quaternion(*camera_heading_idx)
                camera_transform_idx = transformations.calc_transform_matrix(
                    rotation=camera_rotation_idx,
                    position=camera_position_idx
                )
                camera_updated_transform_idx = np.dot(lidar_transform_idx, camera_transform_idx)
                camera_updated_position_idx = transformations.translation_vector_from_transform_matrix(
                    transform_matrix=camera_updated_transform_idx
                )
                camera_updated_rotation_idx = transformations.rotation_matrix_from_transform_matrix(
                    transform_matrix=camera_updated_transform_idx
                )
                camera_updated_heading_idx = transformations.quaternion_from_rotation_matrix(
                    rotation_matrix=camera_updated_rotation_idx
                )

                camera_translation = extrinsic_calibrations.Translation(*camera_updated_position_idx)
                camera_rotation = extrinsic_calibrations.QuaternionRotation(*camera_updated_heading_idx)
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
                    cam_id=camera_id
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
    def build_lidar_scene(lidar_data: dict, cameras_data: dict):
        """
        Merge the all the object of lidar_data and cameras_data to build the LidarScene object that will be uploaded as
        the frames.json item
        :return: scene_data: LidarScene data as JSON that will to be uploaded to the dataloop platform as
        the frames.json item
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

            lidar_scene_frame = lidar_frame.LidarSceneFrame(
                lidar_frame_pcd=frame_lidar_pcd_data,
                lidar_frame_images=lidar_frame_images
            )
            scene.add_frame(lidar_scene_frame)

        scene_data = scene.to_json()
        return scene_data

    # TODO: Check the possibility to use dir Item instead of remote_path
    def run(self, dataset: dl.Dataset, remote_path: str = "/") -> dl.Item:
        """
        Run the parser
        :param dataset: Input dataset
        :param remote_path: Path to the remote folder where the LiDAR data is uploaded
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

            lidar_data = self.parse_lidar_data(items_path=items_path, json_path=json_path)
            cameras_data = self.parse_cameras_data(items_path=items_path, json_path=json_path)
            scene_data = self.build_lidar_scene(lidar_data=lidar_data, cameras_data=cameras_data)

            frames_item = dataset.items.upload(
                remote_name="frames.json",
                remote_path=f"/{remote_path}",
                local_path=json.dumps(scene_data).encode(),
                overwrite=True,
                item_metadata={
                    "system": {
                        "shebang": {
                            "dltype": "PCDFrames"
                        }
                    },
                    "fps": 1
                }
            )
        finally:
            shutil.rmtree(path=download_path, ignore_errors=True)

        return frames_item


def run_parser():
    dataset = dl.datasets.get(dataset_id="6796249cee0196de840650ef")
    parser = AdvancedBaseParser()
    frames_item = parser.run(dataset=dataset, remote_path="/001")
    # frames_item.open_in_web()
    print(frames_item)


if __name__ == '__main__':
    run_parser()
