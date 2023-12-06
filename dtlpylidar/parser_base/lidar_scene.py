from . import lidar_frame
from . import camera_calibrations
from . import extrinsic_calibrations
from typing import List


class LidarScene:
    def __init__(self, metadata=None):
        """
        Lidar Scene object.
        :param metadata: metadata optional.
        """
        self.frames: List[lidar_frame.LidarSceneFrame] = list()
        self.cameras: List[camera_calibrations.LidarCameraData] = list()
        if metadata is None:
            metadata = dict()
        self.metadata = metadata

    def to_json(self):
        """
        Lidar Scene object to dict.
        :return:
        """
        camera_dict = list()
        for camera in self.cameras:
            camera_dict.append(camera.to_json())
        return {
            'isPCDFrames': True,
            'frames': [frame.to_json() for frame in self.frames],
            'cameras': camera_dict,
            'metadata': self.metadata,
            'rotation': extrinsic_calibrations.QuaternionRotation().to_json(),
            'translation': extrinsic_calibrations.Translation().to_json()
        }

    def add_frame(self, frame: lidar_frame.LidarSceneFrame):
        """
        add LidarSceneFrame object to list of frames.
        :param frame: LidarSceneFrame object
        :return:
        """
        self.frames.append(frame)

    def add_camera(self, camera: camera_calibrations.LidarCameraData):
        """
        add LidarCameraData object to list of cameras.
        :param camera: LidarCameraData object
        :return:
        """
        self.cameras.append(camera)
        return camera.cam_id
