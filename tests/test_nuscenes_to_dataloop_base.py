import json
import dtlpy as dl
from dtlpylidar import NuscenesToDataloop, Translation, QuaternionRotation, Intrinsic, Distortion

import unittest
import pandas as pd
import os
import uuid
import shutil


class TestNuscenesToDataloop(unittest.TestCase):
    def setUp(self):
        self.parser = NuscenesToDataloop()
        calibrations_item = dl.items.get(item_id='63f60d330c0faf7a7e0007a8')
        self.csv_path = calibrations_item.download()
        self.calibrations_csv = pd.read_csv(self.csv_path)
        self.folder_item = dl.items.get(item_id='63f5c801c0b5770da1b63d7e')

        self.dataset = self.folder_item.dataset
        filters = dl.Filters()
        filters.add(field=dl.FiltersKnownFields.DIR, values="{}*".format(self.folder_item.filename))
        uid = str(uuid.uuid4())
        base_path = "{}_{}".format(self.dataset.name, uid)
        self.items_download_path = os.path.join(os.getcwd(), base_path)

        # Download items:
        filters.add(field=dl.FiltersKnownFields.NAME, values='*.zip', operator=dl.FILTERS_OPERATIONS_NOT_EQUAL)
        self.dataset.items.download(local_path=self.items_download_path,
                                    filters=filters,
                                    annotation_options=dl.ViewAnnotationOptions.JSON)

        # Path to items and path to jsons.
        self.items_path = os.path.join(self.items_download_path, 'items', self.folder_item.name)
        self.jsons_path = os.path.join(self.items_download_path, 'json', self.folder_item.name)
        # List of paths to all items image folders
        self.image_folders_items = [os.path.join(self.items_path, folder_name) for folder_name in
                                    os.listdir(self.items_path) if 'camera' in folder_name]
        # List of paths to all jsons images folders.
        self.image_folders_jsons = [os.path.join(self.jsons_path, folder_name) for folder_name in
                                    os.listdir(self.jsons_path) if 'camera' in folder_name]

    def test_extract_extrinsic_calibrations(self):
        self.parser.extract_extrinsic_calibrations(remote_dir='098', calibrations_csv=self.calibrations_csv)
        assert len(self.parser.extrinsic_calibrations) == 10
        assert len(self.parser.extrinsic_calibrations[0]) == 7
        assert 'translation' in self.parser.extrinsic_calibrations[0][r'098\camera_front_driver']
        assert 'rotation' in self.parser.extrinsic_calibrations[0][r'098\camera_front_driver']
        assert 'timestamp' in self.parser.extrinsic_calibrations[0][r'098\camera_front_driver']
        assert isinstance(self.parser.extrinsic_calibrations[0][r'098\camera_front_driver']['translation'],
                          Translation)
        assert isinstance(self.parser.extrinsic_calibrations[0][r'098\camera_front_driver']['rotation'],
                          QuaternionRotation)

    def test_extract_intrinsic_calibrations(self):
        self.parser.extract_intrinsic_calibrations(item_name='098', image_folders=self.image_folders_items)
        assert len(self.parser.intrinsic_calibrations) == 10
        assert len(self.parser.intrinsic_calibrations[0]) == 6
        assert 'intrinsic' in self.parser.intrinsic_calibrations[0][r'098\camera_front_driver']
        assert 'distortion' in self.parser.intrinsic_calibrations[0][r'098\camera_front_driver']
        assert isinstance(self.parser.intrinsic_calibrations[0][r'098\camera_front_driver']['intrinsic'],
                          Intrinsic)
        assert isinstance(self.parser.intrinsic_calibrations[0][r'098\camera_front_driver']['distortion'],
                          Distortion)

    def test_parse_frames_local_sensor(self):
        # Extract Extrinsic Calibrations from tf.csv file.
        self.parser.extract_extrinsic_calibrations(remote_dir=self.folder_item.name,
                                                   calibrations_csv=self.calibrations_csv)
        # Extract Intrinsic Calibrations from jsons in camera_info Dir for each camera.
        self.parser.extract_intrinsic_calibrations(item_name=self.folder_item.name,
                                                   image_folders=self.image_folders_items)
        # Sort the images by frame from all the scene cameras
        self.parser.sort_images_by_frame(image_folders=self.image_folders_jsons)
        # Sort PCD Files.
        pcds_path = os.path.join(self.jsons_path, 'velodyne_points')
        if '_' in os.listdir(pcds_path)[0]:
            sorted_pcds = sorted(os.listdir(pcds_path),
                                 key=lambda x: int(x.split('_')[1].split('.')[0]))
        else:
            sorted_pcds = sorted(os.listdir(pcds_path),
                                 key=lambda x: int(x.split('.')[0]))
        self.parser.sorted_pcds = [os.path.join(pcds_path, pcd_file) for pcd_file in sorted_pcds]
        # Parse all the data into DL Lidar Scene
        scene = self.parser.parse_frames_local_sensor(dataset=self.dataset,
                                                      scene_name=self.folder_item.name,
                                                      upload_scene=False)
        with open(os.path.join(self.items_path, 'frames.json'), 'r') as f:
            frames_json_data = json.load(f)
        assert scene.to_json() == frames_json_data

    def tearDown(self):
        print(self.items_download_path)
        shutil.rmtree(self.items_download_path)
