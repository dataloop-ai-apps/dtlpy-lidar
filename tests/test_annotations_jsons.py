import unittest
import os
from dtlpylidar import SampleAnnotation, Sample
import dtlpy as dl
import uuid
import json
import shutil


class TestSampleAnnotation(unittest.TestCase):
    def setUp(self):
        base_path = "{}_{}".format("test_dataset", uuid.uuid4())
        dataset = dl.datasets.get(dataset_id='6388855fd29f985728c126ce')
        self.download_path = os.path.join(os.getcwd(), base_path)
        self.dir_name = 'Duster__2022-10-03-07-39-54_0'
        dataset.items.download(local_path=self.download_path,
                               annotation_options=dl.ViewAnnotationOptions.JSON)
        # annotations download
        filters = dl.Filters()
        annotations_filters = dl.Filters(resource=dl.FiltersResource.ANNOTATION)
        annotations_filters.add(field='type',
                                values='cube_3d')
        path = dataset.download_annotations(local_path=self.download_path,
                                            filters=filters,
                                            annotation_options=dl.ViewAnnotationOptions.JSON,
                                            annotation_filters=annotations_filters)

        self.items_path = os.path.join(path, 'items', self.dir_name)
        self.jsons_path = os.path.join(path, 'json', self.dir_name)
        self.sensor = 'lidar'
        self.frame_num = 0
        self.cam_idx = None
        self.scene_token = 'Duster__2022-10-03-07-39-54_0'
        with open(os.path.join(self.items_path, 'frames.json'), 'r') as f:
            self.lidar_frames = json.load(f).get('frames', dict)
        with open(os.path.join(self.jsons_path, 'frames.json')) as f:
            self.annotation = json.load(f).get('annotations', list)[0]
        sample = Sample(scene_token=self.scene_token,
                        items_path=self.items_path,
                        jsons_path=self.jsons_path,
                        lidar_frames=self.lidar_frames,
                        frame_num=self.frame_num,
                        sensor=self.sensor,
                        cam_idx=self.cam_idx)
        self.sample_data = sample.create_sample_data()

    def test_sample_annotation_initial_value(self):
        sample_annotation = SampleAnnotation(sample_data=self.sample_data, annotation=self.annotation)
        instance_token = "{}_{}".format(self.annotation.get('id'),
                                        self.annotation.get('metadata').get('system').get('objectId'))
        coordinates = self.annotation.get('coordinates')
        translation = [coordinates.get('position').get('x'),
                       coordinates.get('position').get('y'),
                       coordinates.get('position').get('z')]
        size = [coordinates.get('scale').get('x'),
                coordinates.get('scale').get('y'),
                coordinates.get('scale').get('z')]
        rotation = [coordinates.get('rotation').get('x'),
                    coordinates.get('rotation').get('y'),
                    coordinates.get('rotation').get('z')]
        self.assertEqual(sample_annotation.token, '{}_0'.format(self.annotation.get('id')))
        self.assertEqual(sample_annotation.sample_token, self.sample_data.token)
        self.assertListEqual(sample_annotation.attribute_tokens, ['visibility'])
        self.assertEqual(sample_annotation.instance_token, instance_token)
        self.assertListEqual(sample_annotation.translation, translation)
        self.assertListEqual(sample_annotation.size, size)
        self.assertListEqual(sample_annotation.rotation, rotation)
        self.assertEqual(sample_annotation.num_lidar_pts, 5584)
        self.assertEqual(sample_annotation.num_radar_pts, 0)
        self.assertEqual(sample_annotation.next, '{}_1'.format(self.annotation.get('id')))
        self.assertEqual(sample_annotation.prev, '')

    def tearDown(self):
        print(self.download_path)
        shutil.rmtree(self.download_path)


if __name__ == "__main__":
    unittest.main()
