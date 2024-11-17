from dtlpylidar.parser_base import extrinsic_calibrations
from dtlpylidar.parser_base import images_and_pcds, camera_calibrations, lidar_frame, lidar_scene
import dtlpy as dl
import os
import json
from io import BytesIO
import uuid
import logging
import shutil
import pathlib

logger = logging.Logger(name="lidar_base_parser")


class LidarBaseParser(dl.BaseServiceRunner):
    # TODO: Override this method in the derived class
    @staticmethod
    def parse_calibration_data(items_path, json_path) -> dict:
        """
        Extract the calibration data from the downloaded JSONs files
        :param items_path: Path to the items folder
        :param json_path: Path to the JSON folder
        :return: calibration_data: dict with the extracted calibration data
        """
        calibration_data = {"frames": dict()}
        # TODO: change to dynamic
        # os.listdir(json_path)
        camera_list = [
            'front_camera',
            'front_left_camera',
            'left_camera',
            'back_camera',
            'right_camera',
            'front_right_camera'
        ]

        lidar_json_path = os.path.join(json_path, "lidar")
        lidar_items_path = os.path.join(items_path, "lidar")

        # Get all the lidar JSONs sorted by frame number
        lidar_jsons = pathlib.Path(lidar_json_path).rglob('*.json')
        lidar_jsons = sorted(lidar_jsons, key=lambda x: int(x.stem))

        # TODO: read the JSONs files only one
        # store the read data in dict (for all PCD, and Per camera folder - READ every JSON once)
        # Read the ID and filename from the JSONs, instead of passing the path
        for lidar_frame_idx, lidar_json in enumerate(lidar_jsons):
            poses_json = os.path.join(lidar_items_path, "poses.json")
            with open(poses_json, 'r') as f:
                poses_json_data = json.load(f)

            timestamps_json = os.path.join(lidar_items_path, "timestamps.json")
            with open(timestamps_json, 'r') as f:
                timestamps_json_data = json.load(f)

            lidar_data = {
                "path": pathlib.Path(lidar_json).absolute(),
                "timestamp": timestamps_json_data[lidar_frame_idx],
                "position": poses_json_data[lidar_frame_idx]["position"],
                "heading": poses_json_data[lidar_frame_idx]["heading"],
                "cameras": dict()
            }
            calibration_data["frames"][str(lidar_frame_idx)] = lidar_data

            for camera_idx, camera in enumerate(camera_list):
                camera_json_path = os.path.join(json_path, "camera", camera)
                camera_items_path = os.path.join(items_path, "camera", camera)

                camera_json = os.path.join(camera_json_path, pathlib.Path(lidar_json).name)
                camera_frame_idx = int(pathlib.Path(camera_json).stem)

                intrinsics_json = os.path.join(camera_items_path, "intrinsics.json")
                with open(intrinsics_json, 'r') as f:
                    intrinsics_json_data = json.load(f)

                poses_json = os.path.join(camera_items_path, "poses.json")
                with open(poses_json, 'r') as f:
                    poses_json_data = json.load(f)

                timestamps_json = os.path.join(camera_items_path, "timestamps.json")
                with open(timestamps_json, 'r') as f:
                    timestamps_json_data = json.load(f)

                camera_data = {
                    "path": pathlib.Path(camera_json).absolute(),
                    "timestamp": timestamps_json_data[camera_frame_idx],
                    "intrinsics": {
                        "fx": intrinsics_json_data["fx"],
                        "fy": intrinsics_json_data["fy"],
                        "cx": intrinsics_json_data["cx"],
                        "cy": intrinsics_json_data["cy"]
                    },
                    "extrinsics": {
                        "position": poses_json_data[camera_frame_idx]["position"],
                        "heading": poses_json_data[camera_frame_idx]["heading"],
                    },
                    "distortion": {
                        "k1": 0,
                        "k2": 0,
                        "p1": 0,
                        "p2": 0,
                        "k3": 0,
                        "k4": 0
                    }
                }
                calibration_data["frames"][str(lidar_frame_idx)]["cameras"][str(camera_idx)] = camera_data

        return calibration_data

    # TODO: Add to docs to first convert the PLY to PCD
    @staticmethod
    def parse_lidar_data(calibration_data: dict):
        # TODO: Modify to no open the Items JSONs
        """
        Convert the calibration data to a LidarScene object
        :param calibration_data: calibration_data: dict with the extracted calibration data
        :return: buffer: BytesIO buffer with the LidarScene data to be uploaded to the dataloop platform as JSON
        """
        scene = lidar_scene.LidarScene()
        frames = calibration_data.get("frames", dict())
        for frame_num, frame_details in frames.items():
            logger.info(f"Searching PCD {frame_num}")
            pcd_filepath = frame_details.get("path")
            with open(pcd_filepath, 'r') as f:
                pcd_json = json.load(f)

            ground_map_id = pcd_json.get("metadata", dict()).get("user", dict()).get(
                "lidar_ground_detection", dict()).get("groundMapId", None)
            pcd_translation = extrinsic_calibrations.Translation(
                x=frame_details.get("position", dict()).get("x", 0),
                y=frame_details.get("position", dict()).get("y", 0),
                z=frame_details.get("position", dict()).get("z", 0)
            )
            pcd_rotation = extrinsic_calibrations.QuaternionRotation(
                x=frame_details.get("heading", dict()).get("x", 0),
                y=frame_details.get("heading", dict()).get("y", 0),
                z=frame_details.get("heading", dict()).get("z", 0),
                w=frame_details.get("heading", dict()).get("w", 1)
            )
            pcd_time_stamp = frame_details.get("timestamp", "")

            scene_pcd_item = images_and_pcds.LidarPcdData(
                item_id=pcd_json.get("id"),
                ground_id=ground_map_id,
                remote_path=pcd_json.get("filename"),
                extrinsic=extrinsic_calibrations.Extrinsic(
                    rotation=pcd_rotation,
                    translation=pcd_translation
                ),
                timestamp=pcd_time_stamp
            )
            lidar_frame_images = list()
            frame_cameras = frame_details.get("cameras", list())
            for camera_num, camera_details in frame_cameras.items():
                logger.info(f"Searching Image {camera_num} for PCD {frame_num}")
                camera_filepath = camera_details.get("path")
                with open(camera_filepath, 'r') as f:
                    camera_json = json.load(f)

                camera_id = f"{camera_num}_frame_{frame_num}"
                camera_timestamp = camera_details.get("timestamp")
                camera_translation = extrinsic_calibrations.Translation(
                    x=camera_details.get("extrinsics", dict()).get("position").get("x", 0),
                    y=camera_details.get("extrinsics", dict()).get("position").get("y", 0),
                    z=camera_details.get("extrinsics", dict()).get("position").get("z", 0)
                )
                camera_rotation = extrinsic_calibrations.QuaternionRotation(
                    x=camera_details.get("extrinsics", dict()).get("heading").get("x", 0),
                    y=camera_details.get("extrinsics", dict()).get("heading").get("y", 0),
                    z=camera_details.get("extrinsics", dict()).get("heading").get("z", 0),
                    w=camera_details.get("extrinsics", dict()).get("heading").get("w", 1)
                )
                camera_intrinsic = camera_calibrations.Intrinsic(
                    fx=camera_details.get("intrinsics", dict()).get("fx", 0),
                    fy=camera_details.get("intrinsics", dict()).get("fy", 0),
                    cx=camera_details.get("intrinsics", dict()).get("cx", 0),
                    cy=camera_details.get("intrinsics", dict()).get("cy", 0)
                )
                camera_distortion = camera_calibrations.Distortion(
                    k1=camera_details.get("distortion", dict()).get("k1", 0),
                    k2=camera_details.get("distortion", dict()).get("k2", 0),
                    k3=camera_details.get("distortion", dict()).get("k3", 0),
                    p1=camera_details.get("distortion", dict()).get("p1", 0),
                    p2=camera_details.get("distortion", dict()).get("p2", 0)
                )

                lidar_camera = camera_calibrations.LidarCameraData(
                    cam_id=camera_id,
                    intrinsic=camera_intrinsic,
                    extrinsic=extrinsic_calibrations.Extrinsic(
                        rotation=camera_rotation,
                        translation=camera_translation
                    ),
                    channel=camera_details.get("filename"),
                    distortion=camera_distortion
                )

                scene.add_camera(lidar_camera)
                scene_image_item = images_and_pcds.LidarImageData(
                    item_id=camera_json.get("id"),
                    lidar_camera=lidar_camera,
                    remote_path=camera_json.get("filename"),
                    timestamp=camera_timestamp
                )
                lidar_frame_images.append(scene_image_item)

            frame_item = lidar_frame.LidarSceneFrame(
                lidar_frame_pcd=scene_pcd_item,
                lidar_frame_images=lidar_frame_images
            )
            scene.add_frame(frame_item)
        return scene

    @staticmethod
    def download_data(dataset: dl.Dataset, base_path):
        """
        Download the required data for the parser
        :param dataset: Input dataset
        :param base_path: Folder name to store the downloaded data
        :return: download_path: Path to the downloaded data
        """
        download_path = os.path.join(os.getcwd(), base_path)

        # Download items dataloop annotation JSONs
        filters = dl.Filters(field="metadata.system.mimetype", values="*pcd*", method=dl.FiltersMethod.OR)
        filters.add(field="metadata.system.mimetype", values="*image*", method=dl.FiltersMethod.OR)
        dataset.download_annotations(local_path=download_path, filters=filters)

        # Download required binaries (Calibration Data)
        filters = dl.Filters(field="metadata.system.mimetype", values="*json*")
        dataset.items.download(local_path=download_path, filters=filters)

        return download_path

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
        try:
            download_path = self.download_data(dataset=dataset, base_path=base_path)
            items_path = os.path.join(download_path, "items", remote_path)
            json_path = os.path.join(download_path, "json", remote_path)

            calibration_data = self.parse_calibration_data(items_path=items_path, json_path=json_path)
            scene_data = self.parse_lidar_data(calibration_data=calibration_data)

            buffer = BytesIO()
            buffer.write(json.dumps(scene_data.to_json(), default=lambda x: None).encode())
            buffer.seek(0)
            frames_item = dataset.items.upload(
                remote_name="frames.json",
                remote_path=f"/{remote_path}",
                local_path=buffer,
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
            shutil.rmtree(path=base_path, ignore_errors=True)

        return frames_item


def test_parser():
    dataset = dl.datasets.get(dataset_id="673615818ab4c9a0b0be683e")
    parser = LidarBaseParser()
    print(parser.run(dataset=dataset))


if __name__ == '__main__':
    test_parser()
