import open3d as o3d
from dtlpylidar.parser_base.extrinsic_calibrations import Extrinsic
import dtlpylidar.utilities.transformations as transformations
import numpy as np
import dtlpy as dl


def visualize_pcd(item: dl.Item = None, pcd_local_path=None):
    """
    Visualize point cloud using open3d
    :param item: DL item
    :param pcd_local_path: Local point cloud data file
    :return:
    """
    if pcd_local_path is None and item is None:
        raise Exception("Either item id or local file path must be provided")
    if pcd_local_path is None:
        pcd_local_path = item.download()
    cloud = o3d.io.read_point_cloud(pcd_local_path)
    o3d.visualization.draw_geometries([cloud])


def visualize_pcd_with_ground(ground_points: list, item: dl.Item = None, pcd_local_path=None):
    """
          Visualize pcd with ground.
         :param ground_points: list of indices of ground points detected..
         :param item: DL item mimetype pcd.
         :param pcd_local_path: local path to .pcd file.
         :return:
     """

    if pcd_local_path is None and item is None:
        raise Exception("Either DL pcd item or local file path must be provided")
    if ground_points is None:
        raise Exception("list of ground points must be provided")

    if pcd_local_path is None:
        if not item.system.get('mimetype', '*pcd'):
            raise Exception("item mimetype must be pcd")
        pcd_local_path = item.download()
    cloud = o3d.io.read_point_cloud(pcd_local_path)
    inliers_cloud = cloud.select_by_index(ground_points)
    inliers_cloud.paint_uniform_color([1, 0, 0])
    outlier_cloud = cloud.select_by_index(ground_points, invert=True)
    outlier_cloud.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries([inliers_cloud, outlier_cloud])


def visualize_annotation_on_pcd(extrinsic: Extrinsic, annotation_translation, annotation_rotation,
                                annotation_scale, local_path=None, item: dl.Item = None):
    """
    Visualize 3D cube annotation on point cloud
    :param extrinsic: Extrinsic Object for the Point cloud data (translation and rotation)
    :param annotation_translation: Annotation center. [x, y, z]
    :param annotation_rotation: Annotation rotation. [x, y, z]
    :param annotation_scale: annotation scale along each axis. [x, y, z]
    :param local_path: Local path to pcd file
    :param item: DL item
    :return:
    """
    if item is None and local_path is None:
        raise Exception("Either item id or local file path must be provided")
    cube = transformations.calc_cube_points(
        annotation_translation=annotation_translation,
        annotation_rotation=annotation_rotation,
        annotation_scale=annotation_scale
    )

    if local_path is None:
        local_path = item.download()

    cloud = o3d.io.read_point_cloud(local_path)
    cloud_points = np.asarray(cloud.points)
    translated_cloud_points = transformations.translate_point_cloud(points=cloud_points,
                                                                    translation=extrinsic.translation)
    cube = transformations.rotate_point_cloud(points=cube,
                                              rotation=extrinsic.rotation)
    v3d = o3d.utility.Vector3dVector(translated_cloud_points)
    cloud.points = v3d
    lines = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(cube),
                                    lines=o3d.utility.Vector2iVector(lines))
    line_set.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([cloud, line_set])
