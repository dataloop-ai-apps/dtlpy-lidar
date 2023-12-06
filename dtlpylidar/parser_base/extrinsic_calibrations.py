from scipy.spatial.transform import Rotation as R
import logging

logger = logging.getLogger(name='dtlpylidar')


class Translation:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        """
        Translation vector object.
        :param x: x coordinate
        :param y: y coordinate
        :param z: z coordinate
        """
        self.x = x
        self.y = y
        self.z = z

    def to_json(self):
        """
        Translation object to dict
        :return:
        """
        return {
            'x': self.x,
            'y': self.y,
            'z': self.z
        }

    def get_translation_vec(self):
        """
        return translation object as vector.
        :return:
        """
        return [self.x, self.y, self.z]


class QuaternionRotation:
    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):
        """
        Quaternion rotation vector object
        :param x:
        :param y:
        :param z:
        :param w:
        """
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def to_json(self):
        """
        Quaternion rotation object to dict
        :return:
        """
        return {
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'w': self.w
        }

    def get_rotation_vec(self):
        """
        Quaternion rotation object as vector.
        :return:
        """
        return [self.x, self.y, self.z, self.w]


class EulerRotation:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        """
        Euler's Rotation vector object
        :param x: Rotation on x-axis
        :param y: Rotation on y-axis
        :param z: Rotation on z-axis
        """
        self.x = x
        self.y = y
        self.z = z

    def euler_to_quaternion(self):
        """
        Change Euler's rotation to Quaternion rotation
        :return:
        """
        return R.from_euler('xyz', [self.x, self.y, self.z]).as_quat()

    def to_json(self):
        """
        Euler's rotation object to Quaternion rotation object as dict
        DL Lidar scene json supports Quaternion rotation, thus conversion is needed.
        :return:
        """
        quaternion = self.euler_to_quaternion()
        quaternion_rotation = QuaternionRotation(quaternion[0], quaternion[1], quaternion[2], quaternion[3])
        return quaternion_rotation.to_json()

    def get_rotation_vec(self):
        """
        Euler's rotation object as vector
        :return:
        """
        return [self.x, self.y, self.z]


class Extrinsic:
    def __init__(self, rotation, translation: Translation):
        """
        Extrinsic Calibration object
        :param rotation: Quaternion / Euler Rotation object
        :param translation: Translation Object
        """
        if not isinstance(rotation, QuaternionRotation) and not isinstance(rotation, EulerRotation):
            raise TypeError('rotation must be an instance of QuaternionRotation or EulerRotation')
        self.rotation = rotation
        self.translation = translation

    def to_json(self, translation_key):
        """
        Extrinsic calibration to dict.
        :param translation_key: key for Translation Value (translation / position)
        :return:
        """
        return {
            translation_key: self.translation.to_json(),
            'rotation': self.rotation.to_json(),
        }
