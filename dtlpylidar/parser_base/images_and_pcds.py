from . import extrinsic_calibrations
from . import camera_calibrations


class LidarImageData:
    def __init__(self, item_id, lidar_camera: camera_calibrations.LidarCameraData, remote_path=None,
                 timestamp=None):
        """
        Lidar Image object
        :param item_id: Dl image id.
        :param lidar_camera: Lidar Camera object.
        :param remote_path: DL remote path.
        :param timestamp: time of image capturing
        """
        self.camera_id = lidar_camera.cam_id
        self.timestamp = timestamp
        self.item_id = item_id
        self.extrinsic = lidar_camera.extrinsic
        self.remote_path = remote_path

    def to_json(self):
        """
        Lidar image object to dict.
        :return:
        """
        return {
            'camera_id': str(self.camera_id),
            'image_id': self.item_id,
            'sensorsData': {
                'camera': str(self.camera_id),
                'extrinsic': self.extrinsic.to_json(translation_key='translation')
            },
            'timestamp': self.timestamp,
            'remote_path': self.remote_path
        }


class LidarPcdData:
    def __init__(self, item_id, ground_id, extrinsic: extrinsic_calibrations.Extrinsic = None, remote_path=None,
                 timestamp=None):
        """
        Lidar pcd object
        :param item_id: DL pcd ID
        :param ground_id: DL Ground map ID
        :param extrinsic: Extrinsic object
        :param remote_path: DL pcd path
        :param timestamp: time of scan.
        """
        if extrinsic is None:
            extrinsic = extrinsic_calibrations.Extrinsic(rotation=extrinsic_calibrations.QuaternionRotation(),
                                                         translation=extrinsic_calibrations.Translation())
        self.extrinsic = extrinsic
        self.timestamp = timestamp
        self.item_id = item_id
        self.remote_path = remote_path
        self.ground_id = ground_id

    def to_json(self):
        """
        Lidar pcd object to dict.
        :return:
        """
        return {
            'lidar_pcd_id': self.item_id,
            'timestamp': self.timestamp,
            'remote_path': self.remote_path
        }
