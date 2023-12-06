from dtlpylidar import LidarSceneFrame, Intrinsic, Distortion, LidarCameraData, LidarScene, Extrinsic, EulerRotation, \
    Translation, LidarImageData, LidarPcdData
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
        self.lidar_image = LidarImageData(item_id='1',
                                          lidar_camera=self.lidar_camera_data,
                                          remote_path='/',
                                          timestamp='1')
        self.lidar_pcd = LidarPcdData(item_id='2',
                                      ground_id='2',
                                      extrinsic=self.extrinsic,
                                      remote_path='/',
                                      timestamp='2')
        self.lidar_frame_json = LidarSceneFrame(lidar_frame_pcd=self.lidar_pcd, lidar_frame_images=[self.lidar_image])

    def test_add_frame(self):
        scene = LidarScene()
        assert len(scene.frames) == 0
        scene.add_frame(self.lidar_frame_json)
        assert len(scene.frames) == 1

    def test_add_camera(self):
        scene = LidarScene()
        assert len(scene.cameras) == 0
        scene.add_camera(self.lidar_camera_data)
        assert len(scene.cameras) == 1

    def test_to_json(self):
        scene = LidarScene()
        scene.add_camera(self.lidar_camera_data)
        scene.add_frame(self.lidar_frame_json)
        lidar_scene_json = scene.to_json()
        assert lidar_scene_json['frames'] == [self.lidar_frame_json.to_json()]
        assert lidar_scene_json['cameras'] == [self.lidar_camera_data.to_json()]
