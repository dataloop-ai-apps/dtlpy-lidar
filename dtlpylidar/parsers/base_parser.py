import os
import dtlpy as dl
import json
from io import BytesIO
import uuid
import logging
import shutil

from dtlpylidar.parser_base import (extrinsic_calibrations, images_and_pcds, camera_calibrations, lidar_frame,
                                    lidar_scene)

logger = logging.Logger(name="file_mapping_parser")


class LidarFileMappingParser(dl.BaseServiceRunner):
    def __init__(self):
        self.mapping_data = dict()
        self.dataset = None
        self.jsons_path = ""
        self.absolute_path_search = True

    def parse_lidar_data(self, mapping_item: dl.Item) -> dl.Item:
        scene = lidar_scene.LidarScene()
        frames = self.mapping_data.get("frames", dict())
        for frame_num, frame_details in frames.items():
            logger.info(f"Search PCD {frame_num}")
            if frame_details.get("path").startswith("/"):
                pcd_filepath = os.path.join(self.jsons_path,
                                            frame_details.get("path").lstrip('/'))
            else:
                pcd_filepath = os.path.join(self.jsons_path,
                                            mapping_item.dir.lstrip('/'),
                                            frame_details.get("path"))
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
                logger.info(f"Search image {image_num} for frame {frame_num}")
                if image_details.get("image_path").startswith("/"):
                    image_filepath = os.path.join(self.jsons_path,
                                                  image_details.get("image_path").lstrip('/'))
                else:
                    image_filepath = os.path.join(self.jsons_path,
                                                  mapping_item.dir.lstrip('/'),
                                                  image_details.get("image_path"))

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
                    fx=image_details.get("intrinsics", dict()).get("fx", 1),
                    fy=image_details.get("intrinsics", dict()).get("fy", 1),
                    cx=image_details.get("intrinsics", dict()).get("cx", 0),
                    cy=image_details.get("intrinsics", dict()).get("cy", 0),
                    skew=image_details.get("intrinsics", dict()).get("skew", 0)
                )
                camera_distortion = camera_calibrations.Distortion(
                    **image_details.get("distortion", dict())
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
        buffer.name = "frames.json"
        frames_item = self.dataset.items.upload(
            remote_path="{}".format(mapping_item.dir),
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
        return frames_item

    def parse_data(self, mapping_item: dl.Item) -> dl.Item:
        if "json" not in mapping_item.metadata.get("system", dict()).get("mimetype"):
            raise Exception("Expected item of type json")

        buffer = mapping_item.download(save_locally=False)
        self.mapping_data = json.loads(buffer.getvalue())

        self.dataset = mapping_item.dataset
        uid = str(uuid.uuid4())
        base_dataset_name = self.dataset.name
        items_download_path = os.path.join(os.getcwd(), f"{base_dataset_name}_{uid}".lstrip('/\\'))
        try:
            self.dataset.download_annotations(local_path=items_download_path)
            self.jsons_path = os.path.join(items_download_path, "json")
            frames_item = self.parse_lidar_data(mapping_item=mapping_item)
        finally:
            shutil.rmtree(items_download_path, ignore_errors=True)
        return frames_item


def test_parse_data():
    item_id = "<mapping-item-id>"
    parser = LidarFileMappingParser()
    mapping_item = dl.items.get(item_id=item_id)
    print(parser.parse_data(mapping_item=mapping_item))


if __name__ == '__main__':
    test_parse_data()
