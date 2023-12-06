import unittest
from dtlpylidar import CalibratedSensor, Sample, EgosPose
import dtlpy as dl
import uuid
import os
import json
import shutil


class TestCalibratedSensor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base_path = "{}_{}".format("test_dataset", uuid.uuid4())
        dataset = dl.datasets.get(dataset_id='6388855fd29f985728c126ce')
        download_path = os.path.join(os.getcwd(), base_path)
        cls.download_path = os.path.join(os.getcwd(), base_path)
        dir_name = 'Duster__2022-10-03-07-39-54_0'
        dataset.items.download(local_path=download_path,
                               annotation_options=dl.ViewAnnotationOptions.JSON)
        # annotations download
        filters = dl.Filters()
        annotations_filters = dl.Filters(resource=dl.FiltersResource.ANNOTATION)
        annotations_filters.add(field='type',
                                values='cube_3d')
        path = dataset.download_annotations(local_path=download_path,
                                            filters=filters,
                                            annotation_options=dl.ViewAnnotationOptions.JSON,
                                            annotation_filters=annotations_filters)

        items_path = os.path.join(path, 'items', dir_name)
        cls.items_path = os.path.join(path, 'items', dir_name)
        cls.jsons_path = os.path.join(path, 'json', dir_name)
        cls.sensortype = 'camera'
        cls.scene_token = 'Duster__2022-10-03-07-39-54_0'
        with open(os.path.join(items_path, 'frames.json'), 'r') as f:
            lidar_video_content = json.load(f)
        cls.calibrations = lidar_video_content.get('cameras', list())[0]

    def test_calibrated_sensor_initial_value(self):
        camera_intrinsics = [954.354248046875, 0.0, 878.4251588122424, 0.0, 1194.349609375, 426.9580791388871, 0.0, 0.0,
                             1.0]
        rotation = [0.4934491806544604, -0.4832257505605654, 0.5144570281719214, -0.5082664126923349]
        translation = [0.283807, 0.0768354, -0.472547]
        self.calibrated_sensor = CalibratedSensor(calibrations=self.calibrations,
                                                  sensortype=self.sensortype,
                                                  scene_token=self.scene_token,
                                                  items_path=self.items_path,
                                                  jsons_path=self.jsons_path,
                                                  sensor_token='')

        self.assertListEqual(self.calibrated_sensor.camera_intrinsic, camera_intrinsics)
        self.assertListEqual(self.calibrated_sensor.rotation, rotation)
        self.assertListEqual(self.calibrated_sensor.translation, translation)
        self.assertEqual(self.calibrated_sensor.token, 'Duster__2022-10-03-07-39-54_0_0')

    @classmethod
    def tearDownClass(cls):
        print(cls.download_path)
        shutil.rmtree(cls.download_path)


class TestSample(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base_path = "{}_{}".format("test_dataset", uuid.uuid4())
        dataset = dl.datasets.get(dataset_id='6388855fd29f985728c126ce')
        download_path = os.path.join(os.getcwd(), base_path)
        cls.download_path = os.path.join(os.getcwd(), base_path)
        dir_name = 'Duster__2022-10-03-07-39-54_0'
        cls.dir_name = 'Duster__2022-10-03-07-39-54_0'
        dataset.items.download(local_path=download_path,
                               annotation_options=dl.ViewAnnotationOptions.JSON)
        # annotations download
        filters = dl.Filters()
        annotations_filters = dl.Filters(resource=dl.FiltersResource.ANNOTATION)
        annotations_filters.add(field='type',
                                values='cube_3d')
        path = dataset.download_annotations(local_path=download_path,
                                            filters=filters,
                                            annotation_options=dl.ViewAnnotationOptions.JSON,
                                            annotation_filters=annotations_filters)

        items_path = os.path.join(path, 'items', dir_name)
        jsons_path = os.path.join(path, 'json', dir_name)
        sensor = 'lidar'
        frame_num = 0
        cam_idx = None
        scene_token = 'Duster__2022-10-03-07-39-54_0'
        with open(os.path.join(items_path, 'frames.json'), 'r') as f:
            lidar_frames = json.load(f).get('frames', dict())
        cls.sample = Sample(scene_token=scene_token,
                            items_path=items_path,
                            jsons_path=jsons_path,
                            lidar_frames=lidar_frames,
                            frame_num=frame_num,
                            sensor=sensor,
                            cam_idx=cam_idx)

    def test_sample_initial_value(self):
        timestamp = None
        token = '6388857dbe6a8418c2fcb2a3'
        prev = ''
        next = '6388857dbe6a8421adfcb226'
        self.assertEqual(self.sample.token, token)
        self.assertEqual(self.sample.timestamp, timestamp)
        self.assertEqual(self.sample.prev, prev)
        self.assertEqual(self.sample.next, next)

    def test_create_ego_pose(self):
        translation = [0, 0, 0]
        rotation = [0, 0, 0, 1]
        timestamp = None
        ego_pose = self.sample.create_ego_pose()
        self.assertIsInstance(ego_pose, EgosPose)
        self.assertListEqual(ego_pose.translation, translation)
        self.assertListEqual(ego_pose.rotation, rotation)
        self.assertEqual(ego_pose.timestamp, timestamp)

    def test_create_sample_data(self):
        timestamp = None
        token = '6388857dbe6a8418c2fcb2a3_0'
        prev = ''
        next = '6388857dbe6a8421adfcb226_1'
        ego_pose_token = '{}_6388857dbe6a8418c2fcb2a3'.format(self.dir_name)
        calibrated_sensor_token = '{}_lidar'.format(self.dir_name)
        filename = r'/Duster__2022-10-03-07-39-54_0/velodyne_points/1664762995376892075_0.pcd'
        fileformat = 'pcd'
        width = 0
        height = 0
        is_key_frame = False
        sample_data = self.sample.create_sample_data()
        if self.sample.frame_num % 5 == 0:
            is_key_frame = True
        self.assertEqual(sample_data.timestamp, timestamp)
        self.assertEqual(sample_data.prev, prev)
        self.assertEqual(sample_data.token, token)
        self.assertEqual(sample_data.next, next)
        self.assertEqual(sample_data.ego_pose_token, ego_pose_token)
        self.assertEqual(sample_data.calibrated_sensor_token, calibrated_sensor_token)
        self.assertEqual(sample_data.filename, filename)
        self.assertEqual(sample_data.fileformat, fileformat)
        self.assertEqual(sample_data.width, width)
        self.assertEqual(sample_data.height, height)
        self.assertEqual(sample_data.is_key_frame, is_key_frame)

    @classmethod
    def tearDownClass(cls):
        print(cls.download_path)
        shutil.rmtree(cls.download_path)


if __name__ == "__main__":
    unittest.main()
