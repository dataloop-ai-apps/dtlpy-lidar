import unittest
from dtlpylidar import Runner
import dtlpy as dl


class TestRunner(unittest.TestCase):
    def setUp(self):
        self.runner = Runner()
        self.dataset = dl.datasets.get(dataset_id='6388855fd29f985728c126ce')

    def test_create_attribute_file(self):
        attributes = self.runner.create_attribute_file(dataset=self.dataset, scenes_information={})
        # Expecting a single attribute in the recipe with id Visibility.
        self.assertIsInstance(attributes, list)
        assert len(attributes) == 1
        assert attributes[0].get('token') == 'visibility'


if __name__ == "__main__":
    unittest.main()
