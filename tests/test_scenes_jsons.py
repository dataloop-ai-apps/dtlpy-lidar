import unittest
from dtlpylidar import Scene
import dtlpy as dl
import uuid
import os
import json
import shutil


class TestScenes(unittest.TestCase):
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
        try:
            with open(os.path.join(self.items_path, 'scenes_information.json'), 'r') as f:
                self.scenes_information = json.load(f)
        except (FileNotFoundError, IOError):
            self.scenes_information = {}
        self.scene_description = self.scenes_information.get('scene description', 'N/A')

    def test_scene_initial_value(self):
        scene = Scene(dir_name=self.dir_name,
                      items_path=self.items_path,
                      jsons_path=self.jsons_path,
                      scene_description=self.scene_description,
                      log_token='')
        first_pcd_id = '6388857dbe6a8418c2fcb2a3'
        last_pcd_id = '638885889c80ff36ccbd76a4'
        self.assertEqual(scene.nbr_samples, 39)
        self.assertEqual(scene.first_sample_token, first_pcd_id)
        self.assertEqual(scene.last_sample_token, last_pcd_id)

    def tearDown(self):
        print(self.download_path)
        shutil.rmtree(self.download_path)


if __name__ == "__main__":
    unittest.main()
