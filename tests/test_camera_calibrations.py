from dtlpylidar import Intrinsic, Distortion, LidarCameraData, EulerRotation, Translation, Extrinsic
import unittest


class TestLidarCameraData(unittest.TestCase):
    def setUp(self):
        self.intrinsic = Intrinsic()
        self.distortion = Distortion()
        self.extrinsic = Extrinsic(rotation=EulerRotation(),
                                   translation=Translation())
        self.lidar_camera_data = LidarCameraData(intrinsic=self.intrinsic,
                                                 distortion=self.distortion,
                                                 extrinsic=self.extrinsic)

    def test_to_json(self):
        lidar_camera_data_json = self.lidar_camera_data.to_json()
        assert lidar_camera_data_json['name'] == 'cam_0'
        assert lidar_camera_data_json['id'] == '0'
        assert lidar_camera_data_json['sensorsData']['extrinsic'] == self.extrinsic.to_json(translation_key='position')
        assert lidar_camera_data_json['sensorsData']['intrinsicData'] == self.intrinsic.to_json()
        assert lidar_camera_data_json['sensorsData']['intrinsicMatrix'] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        assert 'distortion' in lidar_camera_data_json['sensorsData']['intrinsicData']
        assert 'translation' not in lidar_camera_data_json['sensorsData']['extrinsic']


if __name__ == "__main__":
    unittest.main()
