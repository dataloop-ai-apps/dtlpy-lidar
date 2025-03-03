import unittest
from dtlpylidar.parsers.base_parser import LidarFileMappingParser
import dtlpy as dl



class TestBaseParser(unittest.TestCase):
    def setUp(self):
        dataset_id = "6773c1d24ad8a53aab569c76"
        dataset = dl.datasets.get(dataset_id=dataset_id)

        # Clean dataset
        filters = dl.Filters()
        dataset.items.delete(filters=filters)

        # Prepare dataset
        dataset.items.upload(local_path='assets/example_scene/*')
        self.mapping_item = dataset.items.get(filepath='/mapping.json')


    def test_base_parser(self):
        frames_item = LidarFileMappingParser().parse_data(mapping_item=self.mapping_item)
        self.assertEqual(first=isinstance(frames_item, dl.Item), second=True)


    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
