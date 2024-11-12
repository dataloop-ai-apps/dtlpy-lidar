from typing_extensions import override

from dtlpylidar.parser_base import extrinsic_calibrations
from dtlpylidar.parser_base import images_and_pcds, camera_calibrations, lidar_frame, lidar_scene
import os
import dtlpy as dl
import json
from io import BytesIO
import uuid
import logging
import shutil

logger = logging.Logger(name="file_base_parser")


class LidarBaseParser(dl.BaseServiceRunner):
    @staticmethod
    def sort_pcds_by_frames(**kwargs) -> list:
        pass

    @staticmethod
    def sort_cameras_by_frames(**kwargs) -> list:
        pass

    # TODO: Override this method in the derived class
    def parse_calibration_data(self, dataset: dl.Dataset, remote_path: str) -> dict:
        calibration_data = dict()
        pcds_sorted_data = self.sort_pcds_by_frames(dataset=dataset, remote_path=remote_path)
        images_sorted_data = self.sort_cameras_by_frames(dataset=dataset, remote_path=remote_path)
        return calibration_data

    # TODO: remove jsons_path as it will be used in parse_calibration_data
    @staticmethod
    def parse_lidar_data(jsons_path: str, calibration_data: dict):
        scene = lidar_scene.LidarScene()
        frames = calibration_data.get("frames", dict())
        for frame_num, frame_details in frames.items():
            logger.info(f"Searching PCD {frame_num}")
            pcd_filepath = os.path.join(jsons_path, frame_details.get("path"))
            pcd_filepath = pcd_filepath.replace(".pcd", ".json")
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
            frame_images = frame_details.get("images", list())
            for image_num, image_details in frame_images.items():
                logger.info(f"Searching Image {image_num} for PCD {frame_num}")
                image_filepath = os.path.join(jsons_path, image_details.get("image_path"))
                image_ext = os.path.splitext(image_filepath)[1]
                image_filepath = image_filepath.replace(image_ext, ".json")
                with open(image_filepath, 'r') as f:
                    image_json = json.load(f)

                camera_id = f"{image_num}_frame_{frame_num}"
                image_timestamp = image_details.get("timestamp")
                camera_translation = extrinsic_calibrations.Translation(
                    x=image_details.get("extrinsics", dict()).get("translation").get("x", 0),
                    y=image_details.get("extrinsics", dict()).get("translation").get("y", 0),
                    z=image_details.get("extrinsics", dict()).get("translation").get("z", 0)
                )
                camera_rotation = extrinsic_calibrations.QuaternionRotation(
                    x=image_details.get("extrinsics", dict()).get("rotation").get("x", 0),
                    y=image_details.get("extrinsics", dict()).get("rotation").get("y", 0),
                    z=image_details.get("extrinsics", dict()).get("rotation").get("z", 0),
                    w=image_details.get("extrinsics", dict()).get("rotation").get("w", 1)
                )
                camera_intrinsic = camera_calibrations.Intrinsic(
                    fx=image_details.get("intrinsics", dict()).get("fx", 0),
                    fy=image_details.get("intrinsics", dict()).get("fy", 0),
                    cx=image_details.get("intrinsics", dict()).get("cx", 0),
                    cy=image_details.get("intrinsics", dict()).get("cy", 0)
                )
                camera_distortion = camera_calibrations.Distortion(
                    k1=image_details.get("distortion", dict()).get("k1", 0),
                    k2=image_details.get("distortion", dict()).get("k2", 0),
                    k3=image_details.get("distortion", dict()).get("k3", 0),
                    p1=image_details.get("distortion", dict()).get("p1", 0),
                    p2=image_details.get("distortion", dict()).get("p2", 0)
                )

                lidar_camera = camera_calibrations.LidarCameraData(
                    cam_id=camera_id,
                    intrinsic=camera_intrinsic,
                    extrinsic=extrinsic_calibrations.Extrinsic(
                        rotation=camera_rotation,
                        translation=camera_translation
                    ),
                    channel=image_details.get("image_path"),
                    distortion=camera_distortion
                )

                scene.add_camera(lidar_camera)
                scene_image_item = images_and_pcds.LidarImageData(
                    item_id=image_json.get("id"),
                    lidar_camera=lidar_camera,
                    remote_path=image_json.get("filename"),
                    timestamp=image_timestamp
                )
                lidar_frame_images.append(scene_image_item)

            frame_item = lidar_frame.LidarSceneFrame(
                lidar_frame_pcd=scene_pcd_item,
                lidar_frame_images=lidar_frame_images
            )
            scene.add_frame(frame_item)
        buffer = BytesIO()
        buffer.write(json.dumps(scene.to_json(), default=lambda x: None).encode())
        buffer.seek(0)
        return buffer

    def run(self, dataset: dl.Dataset, remote_path: str = "/"):
        if remote_path.startswith("/"):
            remote_path = remote_path[1:]

        if remote_path.endswith("/"):
            remote_path = remote_path[:-1]

        base_path = f"{dataset.name}_{str(uuid.uuid4())}"
        try:
            download_path = os.path.join(os.getcwd(), base_path)
            dataset.download_annotations(local_path=download_path)
            jsons_path = os.path.join(download_path, "json", remote_path)

            calibration_data = self.parse_calibration_data(
                dataset=dataset,
                remote_path=f"/{remote_path}/"
            )
            buffer = self.parse_lidar_data(
                jsons_path=jsons_path,
                calibration_data=calibration_data
            )
            frames_item = dataset.items.upload(
                remote_name="frames.json",
                remote_path=f"/{remote_path}/",
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
            shutil.rmtree(base_path, ignore_errors=True)

        return frames_item



def test_parse_data():
    dataset = dl.datasets.get(dataset_id="<dataset-id>")
    parser = LidarBaseParser()
    print(parser.run(dataset=dataset))


if __name__ == '__main__':
    test_parse_data()
