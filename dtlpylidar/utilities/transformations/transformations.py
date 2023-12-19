import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from dtlpylidar.parser_base.extrinsic_calibrations import Translation, QuaternionRotation, EulerRotation
import math


def rotation_matrix_from_euler(rotation_x, rotation_y, rotation_z):
    """
    Calculate rotation matrix from euler angles (x,y,z)
    :param rotation_x:
    :param rotation_y:
    :param rotation_z:
    :return:
    """
    return R.from_euler('xyz', [rotation_x, rotation_y, rotation_z]).as_matrix()


def rotation_matrix_from_quaternion(rotation_x, rotation_y, rotation_z, rotation_w):
    """
    Calculate rotation matrix from quaternion angles (x,y,z,w)
    :param rotation_x:
    :param rotation_y:
    :param rotation_z:
    :param rotation_w:
    :return:
    """
    return R.from_quat([rotation_x, rotation_y, rotation_z, rotation_w]).as_matrix()


def euler_from_quaternion(rotation_x, rotation_y, rotation_z, rotation_w):
    """
    Calculate euler rotation (x,y,z) from quaternion angles (x,y,z,w)
    :param rotation_x:
    :param rotation_y:
    :param rotation_z:
    :return:
    """
    return R.from_quat([rotation_x, rotation_y, rotation_z, rotation_w]).as_euler('xyz')


def quaternion_from_euler(rotation_x, rotation_y, rotation_z):
    """
    Calculate quaternion angles (x,y,z,w) from euler rotation (x,y,z)
    :param rotation_x:
    :param rotation_y:
    :param rotation_z:
    :return:
    """
    return R.from_euler('xyz', [rotation_x, rotation_y, rotation_z]).as_quat()


def calc_translation_matrix(x_position=0, y_position=0, z_position=0):
    """
    Calculate translation matrix from x,y,z position
    :param x_position:
    :param y_position:
    :param z_position:
    :return:
    """
    return np.asarray([
        [1, 0, 0, x_position],
        [0, 1, 0, y_position],
        [0, 0, 1, z_position],
        [0, 0, 0, 1]
    ])


def calc_rotation_matrix(theta_x=0, theta_y=0, theta_z=0, radians: bool = True, epsilon=1e-5):
    """
    Calculate rotation matrix from theta_x,theta_y,theta_z angles
    :param theta_x:
    :param theta_y:
    :param theta_z:
    :param radians: True for Radian thetas, False for Degree thetas.
    :param epsilon: Limitation for small numbers close to zero
    :return:
    """
    if radians is False:
        theta_x = math.radians(theta_x) if theta_x else 0
        theta_y = math.radians(theta_y) if theta_y else 0
        theta_z = math.radians(theta_z) if theta_z else 0

    rotation = np.identity(4)
    if theta_x:
        rotation_x = np.array([
            [1, 0, 0, 0],
            [0, math.cos(theta_x), -math.sin(theta_x), 0],
            [0, math.sin(theta_x), math.cos(theta_x), 0],
            [0, 0, 0, 1]
        ])
        rotation = rotation @ rotation_x
    if theta_y:
        rotation_y = np.array([
            [math.cos(theta_y), 0, math.sin(theta_y), 0],
            [0, 1, 0, 0],
            [-math.sin(theta_y), 0, math.cos(theta_y), 0],
            [0, 0, 0, 1]
        ])
        rotation = rotation @ rotation_y
    if theta_z:
        rotation_z = np.array([
            [math.cos(theta_z), -math.sin(theta_z), 0, 0],
            [math.sin(theta_z), math.cos(theta_z), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        rotation = rotation @ rotation_z
    if epsilon is not None:
        rotation[np.abs(rotation) < epsilon] = 0
    return rotation


def calc_transform_matrix(quaternion=np.array([0, 0, 0, 1]), position=np.array([0, 0, 0])):
    rotation = rotation_matrix_from_quaternion(quaternion[0], quaternion[1], quaternion[2], quaternion[3])
    transform_matrix = np.identity(n=4)
    transform_matrix[0: 3, 0: 3] = rotation
    transform_matrix[0: 3, 3] = position
    return transform_matrix


def translate_point_cloud(points, translation: Translation):
    """
    Translate point cloud by x,y,z position
    :param points:
    :param translation:
    :return:
    """
    return points - np.asarray(translation.get_translation_vec())


def rotate_point_cloud(points, rotation):
    """
    Rotate point cloud by euler or quaternion rotation
    :param points:
    :param rotation:
    :return:
    """
    if not isinstance(rotation, EulerRotation) and not isinstance(rotation, QuaternionRotation):
        raise Exception('rotation must be of type Euler or Quaternion rotation')
    rotation_vec = rotation.get_rotation_vec()
    if isinstance(rotation, EulerRotation):
        rotation_matrix = rotation_matrix_from_euler(rotation_x=rotation_vec[0],
                                                     rotation_y=rotation_vec[1],
                                                     rotation_z=rotation_vec[2])
    else:
        rotation_matrix = rotation_matrix_from_quaternion(rotation_x=rotation_vec[0],
                                                          rotation_y=rotation_vec[1],
                                                          rotation_z=rotation_vec[2],
                                                          rotation_w=rotation_vec[3])

    return np.dot(np.linalg.inv(rotation_matrix), points.transpose()).transpose()


def rotate_annotation_cube3d(annotation_corners, rotation_matrix, translation_matrix):
    """
    Rotate cube by euler angles and translation matrix
    :param annotation_corners:
    :param rotation_matrix:
    :param translation_matrix:
    :return:
    """
    points = np.dot(np.linalg.inv(translation_matrix),
                    np.concatenate((annotation_corners, np.ones((annotation_corners.shape[0], 1))),
                                   axis=1).transpose()).transpose()
    points = np.dot(rotation_matrix, points[:, :3].transpose()).transpose()
    points = np.dot(translation_matrix,
                    np.concatenate((points, np.ones((points.shape[0], 1))), axis=1).transpose()).transpose()
    return points[:, :3]


def scale_point_cloud_data(pcd: o3d.geometry.PointCloud, scale_factor: float):
    """
    Scale point cloud data by scale factor
    :param pcd: open3d point cloud
    :param scale_factor: scale factor
    :return:
    """
    pcd.scale(scale_factor, center=pcd.get_center())


def calc_cube_points(annotation_translation, annotation_scale, annotation_rotation=None, apply_rotation=True):
    """
    Given a 3d cube represented by center point, rotation and scale, calculate the 8 corners of the cube
    :param annotation_translation: Annotation center. [x, y, z]
    :param annotation_rotation: Annotation rotation. [x, y, z]
    :param annotation_scale: annotation scale along each axis. [x, y, z]
    :param apply_rotation: Apply rotation to the cube
    :return:
    """
    x_scale = annotation_scale[0]
    y_scale = annotation_scale[1]
    z_scale = annotation_scale[2]

    x_position = annotation_translation[0]
    y_position = annotation_translation[1]
    z_position = annotation_translation[2]

    cube = np.asarray([
        [x_position + x_scale / 2, y_position + y_scale / 2, z_position + z_scale / 2],
        [x_position + x_scale / 2, y_position + y_scale / 2, z_position - z_scale / 2],
        [x_position + x_scale / 2, y_position - y_scale / 2, z_position + z_scale / 2],
        [x_position + x_scale / 2, y_position - y_scale / 2, z_position - z_scale / 2],
        [x_position - x_scale / 2, y_position + y_scale / 2, z_position + z_scale / 2],
        [x_position - x_scale / 2, y_position + y_scale / 2, z_position - z_scale / 2],
        [x_position - x_scale / 2, y_position - y_scale / 2, z_position + z_scale / 2],
        [x_position - x_scale / 2, y_position - y_scale / 2, z_position - z_scale / 2],
    ])

    if apply_rotation is True:
        if annotation_rotation is None:
            raise Exception("Rotation must be provided")
        x_rotation = annotation_rotation[0]
        y_rotation = annotation_rotation[1]
        z_rotation = annotation_rotation[2]
        rotation_matrix = rotation_matrix_from_euler(
            rotation_x=x_rotation,
            rotation_y=y_rotation,
            rotation_z=z_rotation)
        translation_matrix = calc_translation_matrix(
            x_position=x_position,
            y_position=y_position,
            z_position=z_position)
        cube = rotate_annotation_cube3d(annotation_corners=cube,
                                        rotation_matrix=rotation_matrix,
                                        translation_matrix=translation_matrix)
    return cube
