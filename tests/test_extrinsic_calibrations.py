from dtlpylidar import Extrinsic, EulerRotation, QuaternionRotation, Translation
import unittest


class TestExtrinsic(unittest.TestCase):
    def setUp(self):
        self.euler_rotation = EulerRotation()
        self.quaternion_rotation = QuaternionRotation()
        self.translation = Translation()
        self.rotation_list = [0, 0, 0]

    def test_extrinsic_initial_value(self):
        self.assertRaises(TypeError, Extrinsic, (self.rotation_list, self.translation))

    def test_to_json(self):
        extrinsic_euler = Extrinsic(rotation=self.euler_rotation, translation=self.translation)
        extrinsic_quaternion = Extrinsic(rotation=self.quaternion_rotation, translation=self.translation)
        assert extrinsic_quaternion.to_json('translation') == extrinsic_euler.to_json('translation')
        assert 'translation' in extrinsic_quaternion.to_json('translation')
        assert 'translation' not in extrinsic_quaternion.to_json('position')
        assert 'rotation' in extrinsic_quaternion.to_json('translation')


if __name__ == "__main__":
    unittest.main()
