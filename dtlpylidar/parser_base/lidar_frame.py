from . import images_and_pcds
from typing import List


class LidarSceneFrame:
    def __init__(self, lidar_frame_pcd: images_and_pcds.LidarPcdData,
                 lidar_frame_images: List[images_and_pcds.LidarImageData],
                 metadata=None):
        """
        Lidar frame object.
        :param lidar_frame_pcd: LidarPcdData object.
        :param lidar_frame_images: List of LidarImageData objects.
        :param metadata: Optional Metadata.
        """
        if metadata is None:
            metadata = dict()
        self.metadata = metadata
        self.lidar_frame_images = lidar_frame_images
        self.lidar_frame_pcd = lidar_frame_pcd

    def add_image(self, image: images_and_pcds.LidarImageData):
        """
        Add LidarImageData object to frame's list of images.
        :param image:
        :return:
        """
        self.lidar_frame_images.append(image)

    def to_json(self):
        """
        Lidar frame object to dict.
        :return:
        """
        if len(self.lidar_frame_images) > 0:
            images_to_json = [image.to_json() for image in self.lidar_frame_images]
        else:
            images_to_json = []
        return {
            'translation': self.lidar_frame_pcd.extrinsic.translation.to_json(),
            'rotation': self.lidar_frame_pcd.extrinsic.rotation.to_json(),
            'metadata': self.metadata,
            'lidar': self.lidar_frame_pcd.to_json(),
            'images': images_to_json,
            'groundMapId': self.lidar_frame_pcd.ground_id
        }
