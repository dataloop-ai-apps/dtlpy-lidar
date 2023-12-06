# from dtlpylidar import LidarSceneFrame, Intrinsic, Distortion, LidarCameraData, Extrinsic, EulerRotation, Translation, \
#     LidarImageData, LidarPcdData
# from dtlpylidar import Intrinsic, Distortion, LidarCameraData
import dtlpylidar
import unittest


class TestLidarCameraData(unittest.TestCase):
    def setUp(self):
        self.intrinsic = dtlpylidar.Intrinsic()
        self.distortion = dtlpylidar.Distortion()
        self.extrinsic = dtlpylidar.Extrinsic(rotation=dtlpylidar.EulerRotation(),
                                   translation=dtlpylidar.Translation())
        self.lidar_camera_data = dtlpylidar.LidarCameraData(intrinsic=self.intrinsic,
                                                 distortion=self.distortion,
                                                 extrinsic=self.extrinsic)
        self.lidar_image = dtlpylidar.LidarImageData(item_id='1',
                                          lidar_camera=self.lidar_camera_data,
                                          remote_path='/',
                                          timestamp='1')
        self.lidar_pcd = dtlpylidar.LidarPcdData(item_id='2',
                                      ground_id='2',
                                      extrinsic=self.extrinsic,
                                      remote_path='/',
                                      timestamp='2')

    def test_add_image(self):
        lidar_frame_json = dtlpylidar.LidarSceneFrame(lidar_frame_pcd=self.lidar_pcd, lidar_frame_images=[])
        assert lidar_frame_json.lidar_frame_images == []
        assert len(lidar_frame_json.lidar_frame_images) == 0
        assert lidar_frame_json.to_json()['images'] == []
        lidar_frame_json.add_image(image=self.lidar_image)
        assert len(lidar_frame_json.lidar_frame_images) == 1

    def test_to_json(self):
        lidar_frame_json = dtlpylidar.LidarSceneFrame(lidar_frame_pcd=self.lidar_pcd, lidar_frame_images=[self.lidar_image])
        assert lidar_frame_json.to_json()['images'] == [self.lidar_image.to_json()]
        assert lidar_frame_json.to_json()['lidar'] == self.lidar_pcd.to_json()
        assert lidar_frame_json.to_json()['groundMapId'] == '2'
        assert lidar_frame_json.to_json()['translation'] == dtlpylidar.Translation().to_json()
        assert lidar_frame_json.to_json()['rotation'] == dtlpylidar.EulerRotation().to_json()


if __name__ == "__main__":
    unittest.main()
