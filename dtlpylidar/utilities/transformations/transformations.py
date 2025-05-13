import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from dtlpylidar.parser_base.extrinsic_calibrations import Translation, QuaternionRotation, EulerRotation
import math


def rotation_matrix_from_euler(rotation_x=0.0, rotation_y=0.0, rotation_z=0.0, degrees: bool = False, seq: str = "xyz"):
    """
    Calculate rotation matrix from euler angles (x,y,z)
    :param rotation_x: number
    :param rotation_y: number
    :param rotation_z: number
    :param degrees: True for Degree rotations, False for Radian rotations.
    :param seq: Euler angles sequence
    :return: 3x3 rotation matrix
    """
    return R.from_euler(seq=seq, angles=[rotation_x, rotation_y, rotation_z], degrees=degrees).as_matrix()


def euler_from_rotation_matrix(rotation_matrix=np.identity(3), degrees: bool = False, seq: str = "xyz"):
    """
    Calculate euler angles (x,y,z) from rotation matrix
    :param rotation_matrix: 3x3 rotation matrix
    :param degrees: True for Degree euler angles, False for Radian euler angles.
    :param seq: Euler angles sequence
    :return: 3x1 euler angles vector
    """
    return R.from_matrix(rotation_matrix).as_euler(seq=seq, degrees=degrees)


def rotation_matrix_from_quaternion(quaternion_x=0.0, quaternion_y=0.0, quaternion_z=0.0, quaternion_w=1.0):
    """
    Calculate rotation matrix from quaternion angles (x,y,z,w)
    :param quaternion_x: number
    :param quaternion_y: number
    :param quaternion_z: number
    :param quaternion_w: number
    :return: 3x3 rotation matrix
    """
    return R.from_quat([quaternion_x, quaternion_y, quaternion_z, quaternion_w]).as_matrix()


def quaternion_from_rotation_matrix(rotation_matrix=np.identity(3)):
    """
    Calculate quaternion angles (x,y,z,w) from rotation matrix
    :param rotation_matrix: 3x3 rotation matrix
    :return: 4x1 quaternion angles vector
    """
    return R.from_matrix(rotation_matrix).as_quat()


def euler_from_quaternion(quaternion_x=0.0, quaternion_y=0.0, quaternion_z=0.0, quaternion_w=1.0,
                          degrees: bool = False, seq: str = "xyz"):
    """
    Calculate euler angles (x,y,z) from quaternion angles (x,y,z,w)
    :param quaternion_x: number
    :param quaternion_y: number
    :param quaternion_z: number
    :param quaternion_w: number
    :param degrees: True for Degree euler angles, False for Radian euler angles.
    :param seq: Euler angles sequence
    :return: 3x1 euler angles vector
    """
    return R.from_quat([quaternion_x, quaternion_y, quaternion_z, quaternion_w]).as_euler(seq=seq, degrees=degrees)


def quaternion_from_euler(rotation_x=0.0, rotation_y=0.0, rotation_z=0.0, degrees: bool = False, seq: str = "xyz"):
    """
    Calculate quaternion angles (x,y,z,w) from euler angles (x,y,z)
    :param rotation_x: number
    :param rotation_y: number
    :param rotation_z: number
    :param degrees: True for Degree rotations, False for Radian rotations.
    :param seq: Euler angles sequence
    :return: 4x1 quaternion angles vector
    """
    return R.from_euler(seq=seq, angles=[rotation_x, rotation_y, rotation_z], degrees=degrees).as_quat()


def translation_vector_from_transform_matrix(transform_matrix=np.identity(4)):
    """
    Extract position (x,y,z) from transform matrix
    :param transform_matrix: 4x4 transform matrix
    :return: 3x1 translation vector
    """
    return transform_matrix[0: 3, 3]


def rotation_matrix_from_transform_matrix(transform_matrix=np.identity(4)):
    """
    Extract rotation matrix from transform matrix
    :param transform_matrix: 4x4 transform matrix
    :return: 3x3 rotation matrix
    """
    return transform_matrix[0: 3, 0: 3]


def calc_translation_matrix(position_x=0.0, position_y=0.0, position_z=0.0):
    """
    Calculate translation matrix from position (x,y,z)
    :param position_x: number
    :param position_y: number
    :param position_z: number
    :return: 4x4 translation matrix
    """
    return np.asarray([
        [1, 0, 0, position_x],
        [0, 1, 0, position_y],
        [0, 0, 1, position_z],
        [0, 0, 0, 1]
    ])


def calc_rotation_matrix(theta_x=0.0, theta_y=0.0, theta_z=0.0, degrees: bool = True):
    """
    Calculate rotation matrix from theta angles (x,y,z)
    :param theta_x: number
    :param theta_y: number
    :param theta_z: number
    :param degrees: True for Degree thetas, False for Radian thetas.
    :return: 3x3 rotation matrix
    """
    if degrees is True:
        theta_x = math.radians(theta_x) if theta_x != 0.0 else 0.0
        theta_y = math.radians(theta_y) if theta_y != 0.0 else 0.0
        theta_z = math.radians(theta_z) if theta_z != 0.0 else 0.0

    rotation = np.identity(3)
    if theta_x != 0.0:
        rotation_x = np.array([
            [1.0, 0.0, 0.0],
            [0.0, math.cos(theta_x), -math.sin(theta_x)],
            [0.0, math.sin(theta_x), math.cos(theta_x)]
        ])
        rotation = rotation @ rotation_x
    if theta_y != 0.0:
        rotation_y = np.array([
            [math.cos(theta_y), 0.0, math.sin(theta_y)],
            [0.0, 1.0, 0.0],
            [-math.sin(theta_y), 0.0, math.cos(theta_y)]
        ])
        rotation = rotation @ rotation_y
    if theta_z != 0.0:
        rotation_z = np.array([
            [math.cos(theta_z), -math.sin(theta_z), 0.0],
            [math.sin(theta_z), math.cos(theta_z), 0.0],
            [0.0, 0.0, 1.0]
        ])
        rotation = rotation @ rotation_z
    return rotation


def calc_transform_matrix(rotation=np.identity(n=3), position=np.zeros(3), seq: str = "xyz"):
    """
    Calculate transform matrix from rotation matrix and position
    :param rotation: 3x3 - Rotation matrix, 3x1 - Euler angles (with seq) or 4x1 - Quaternion
    :param position: 3x1 - Translation vector
    :param seq: Euler angles sequence
    :return: 4x4 transform matrix
    """
    if rotation.shape == (3, 3):  # Rotation matrix
        pass  # No change needed
    elif rotation.shape == (3, ):  # Euler angles sequence
        rotation = rotation_matrix_from_euler(*rotation, seq=seq)
    elif rotation.shape == (4, ):  # Quaternion
        rotation = rotation_matrix_from_quaternion(*rotation)
    else:
        raise Exception('Shape of rotation matrix is not valid. Must be 3x3, 3x1 or 4x1')

    transform_matrix = np.identity(n=4)
    transform_matrix[0: 3, 0: 3] = rotation
    transform_matrix[0: 3, 3] = position
    return transform_matrix


def apply_translation(transform_matrix=np.identity(4), translation_vector=np.zeros(3)):
    """
    Apply translation matrix to a 4x4 transformation matrix.
    :param transform_matrix: 4x4 transform matrix
    :param translation_vector: 3x1 translation vector
    :return: 4x4 transform matrix
    """
    new_transform = transform_matrix.copy()
    new_transform[:3, 3] += translation_vector
    return new_transform


def apply_rotation(transform_matrix=np.identity(4), rotation_matrix=np.identity(3),
                   from_right=True, rotate_around=None):
    """
    Apply rotation matrix to a 4x4 transformation matrix.
    :param transform_matrix: 4x4 transform matrix
    :param rotation_matrix: 3x3 rotation matrix
    :param from_right: True to apply rotation from right, False to apply rotation from left
    :param rotate_around: 3x1 vector representing the point to rotate around
    (default is None, which means rotate around the object's center).
    :return: 4x4 transform matrix
    """
    new_transform = transform_matrix.copy()
    rotation_transform = calc_transform_matrix(rotation=rotation_matrix, position=np.zeros(3))

    # Rotate around a specific point
    if rotate_around is not None:
        translation = new_transform[:3, 3]
        direction_vector = np.array(rotate_around) - translation

        # Translate object so that 'rotate_around' becomes the origin
        new_transform = apply_translation(new_transform, direction_vector)

        if from_right is True:
            new_transform = np.dot(new_transform, rotation_transform)
        else:
            new_transform = np.dot(rotation_transform, new_transform)

        # Rotate the direction vector and translate back
        rotated_direction_vector = np.dot(rotation_matrix, direction_vector)
        new_transform = apply_translation(new_transform, rotated_direction_vector)
    else:
        # Rotate around the object's center
        if from_right is True:
            new_transform = np.dot(new_transform, rotation_transform)
        else:
            new_transform = np.dot(rotation_transform, new_transform)

    return new_transform


def apply_translation_in_rotation_direction(transform_matrix=np.identity(4),
                                            rotation_matrix=np.identity(3), translation_vector=np.zeros(3)):
    """
    Apply translation in the direction of the rotation matrix to a 4x4 transformation matrix.
    :param transform_matrix: 4x4 transform matrix
    :param rotation_matrix: 3x3 rotation matrix
    :param translation_vector: 3x1 translation vector
    :return: 4x4 transform matrix
    """
    new_transform = transform_matrix.copy()

    rotated_direction_vector = np.dot(rotation_matrix, translation_vector)
    new_transform = apply_translation(new_transform, rotated_direction_vector)
    return new_transform


def calc_cuboid_corners(center=np.zeros(3), dimensions=np.ones(3)):
    """
    Calculates the 3D coordinates of all eight corners of a cube given its center and dimensions.

    Args:
        center: A list or numpy array of size 3 representing the center point (x, y, z).
        dimensions: A list or numpy array of size 3 representing the dimensions (x, y, z) of the cube.

    Returns:
        A numpy array of size (8, 3) containing the coordinates of all eight corners.
    """

    # Half dimensions for easier calculations
    half_dimensions = np.array(dimensions) / 2

    # Create a list of corner offsets relative to the center
    corner_offsets = [
        [-1, -1, -1], [+1, -1, -1], [-1, +1, -1], [+1, +1, -1],
        [-1, -1, +1], [+1, -1, +1], [-1, +1, +1], [+1, +1, +1]
    ]

    # Convert corner offsets to numpy array
    corner_offsets = np.array(corner_offsets)

    # Calculate corner positions by adding offsets to center and multiplying by half dimensions
    cuboid_corners = center + corner_offsets * half_dimensions
    return cuboid_corners


def fix_cuboid_directions(cuboid_corners: np.ndarray, cuboid_rotation_matrix=np.identity(3),
                          scene_position=np.zeros(3), scene_rotation_matrix=np.identity(3)):
    scene_transformation_matrix = calc_transform_matrix(
        rotation=scene_rotation_matrix,
        position=scene_position
    )
    v3d = o3d.utility.Vector3dVector(cuboid_corners)
    cloud = o3d.geometry.PointCloud(v3d)
    cloud.transform(scene_transformation_matrix)
    new_cuboid_rotation = np.dot(scene_rotation_matrix, cuboid_rotation_matrix)
    new_cuboid_position = cloud.get_center() - scene_position
    return new_cuboid_position, new_cuboid_rotation


def calc_cuboid_scene_transform_matrix(cuboid_position=np.zeros(3), cuboid_quaternion=np.asarray([0.0, 0.0, 0.0, 1.0]),
                                       cuboid_scale=np.ones(3),
                                       scene_position=np.zeros(3), scene_quaternion=np.asarray([0.0, 0.0, 0.0, 1.0])):
    cuboid_rotation_matrix = rotation_matrix_from_quaternion(*cuboid_quaternion)
    cuboid_corners = calc_cuboid_corners(center=cuboid_position, dimensions=cuboid_scale)
    scene_rotation_matrix = rotation_matrix_from_quaternion(*scene_quaternion)

    new_cuboid_position, new_cuboid_rotation = fix_cuboid_directions(
        cuboid_corners=cuboid_corners,
        cuboid_rotation_matrix=cuboid_rotation_matrix,
        scene_position=scene_position,
        scene_rotation_matrix=scene_rotation_matrix,
    )
    new_cuboid_transform_matrix = calc_transform_matrix(rotation=new_cuboid_rotation, position=new_cuboid_position)
    return new_cuboid_transform_matrix


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
        rotation_matrix = rotation_matrix_from_quaternion(quaternion_x=rotation_vec[0],
                                                          quaternion_y=rotation_vec[1],
                                                          quaternion_z=rotation_vec[2],
                                                          quaternion_w=rotation_vec[3])

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


def transform_points(points: np.ndarray, translation: np.ndarray, rotation: np.ndarray):
    # Step 1: Convert points to homogeneous coordinates (Nx4)
    ones = np.ones((points.shape[0], 1))
    points_hom = np.hstack([points, ones])  # Nx4

    # Step 2: Build cube's transformation matrix and apply inverse to move points to cube-local frame
    transform_matrix = calc_transform_matrix(
        rotation=rotation,
        position=translation
    )
    transformed_hom = (transform_matrix @ points_hom.T).T  # still Nx4
    transformed_points = transformed_hom[:, :3]
    return transformed_points


def calc_cube_points(annotation_translation=np.array([0.0, 0.0, 0.0]),
                     annotation_scale=np.array([1.0, 1.0, 1.0]),
                     annotation_rotation=np.array([0.0, 0.0, 0.0])):
    """
    Given a 3d cube represented by center point, rotation and scale, calculate the 8 corners of the cube
    :param annotation_translation: Annotation center. [x, y, z]
    :param annotation_scale: annotation scale along each axis. [x, y, z]
    :param annotation_rotation: Annotation rotation. [x, y, z]
    :return:
    """
    if isinstance(annotation_translation, list):
        annotation_translation = np.array(annotation_translation)
    if isinstance(annotation_scale, list):
        annotation_scale = np.array(annotation_scale)
    if isinstance(annotation_rotation, list):
        annotation_rotation = np.array(annotation_rotation)

    # Center of the cube
    position_x = 0
    position_y = 0
    position_z = 0

    # Scale of the cube
    scale_x = annotation_scale[0]
    scale_y = annotation_scale[1]
    scale_z = annotation_scale[2]

    cube_points = np.asarray([
        [position_x + scale_x / 2, position_y + scale_y / 2, position_z + scale_z / 2],
        [position_x + scale_x / 2, position_y + scale_y / 2, position_z - scale_z / 2],
        [position_x + scale_x / 2, position_y - scale_y / 2, position_z + scale_z / 2],
        [position_x + scale_x / 2, position_y - scale_y / 2, position_z - scale_z / 2],
        [position_x - scale_x / 2, position_y + scale_y / 2, position_z + scale_z / 2],
        [position_x - scale_x / 2, position_y + scale_y / 2, position_z - scale_z / 2],
        [position_x - scale_x / 2, position_y - scale_y / 2, position_z + scale_z / 2],
        [position_x - scale_x / 2, position_y - scale_y / 2, position_z - scale_z / 2],
    ])

    cube_points = transform_points(
        points=cube_points,
        translation=annotation_translation,
        rotation=annotation_rotation
    )
    return cube_points
