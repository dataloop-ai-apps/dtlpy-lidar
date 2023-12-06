import json
import os
import open3d as o3d
import numpy as np


class Instances:
    def __init__(self):
        self.instances_output = dict()

    def add_instance(self, instances):
        """
        add instances list to the instances_output dict
        :param instances:
        :return:
        """
        for instance in instances:
            self.instances_output[instance.token] = instance

    def to_json(self, output_path):
        instance_json = list()
        for instance_token, instance in self.instances_output.items():
            instance_json.append({
                'token': instance_token,
                'category_token': instance.category_token,
                'nbr_annotations': instance.nbr_annotations,
                'first_annotation_token': instance.first_annotation_token,
                'last_annotation_token': instance.last_annotation_token
            })
        instance_path = os.path.join(output_path, 'instance.json')
        with open(instance_path, 'w') as f:
            json.dump(instance_json, f, indent=4)


class Instance:
    def __init__(self, token, category_token, nbr_annotations, first_annotation_token, last_annotation_token):
        self.token = token
        self.category_token = category_token
        self.nbr_annotations = nbr_annotations
        self.first_annotation_token = first_annotation_token
        self.last_annotation_token = last_annotation_token


# Sample annotations
class SampleAnnotations:
    def __init__(self):
        self.sample_annotations_output = dict()

    def add_sample_annotation(self, sample_annotations):
        for sample_annotation in sample_annotations:
            self.sample_annotations_output[sample_annotation.token] = sample_annotation

    def to_json(self, output_path):
        sample_annotation_json = list()
        for sample_annotation_token, sample_annotation in self.sample_annotations_output.items():
            sample_annotation_json.append({
                'token': sample_annotation_token,
                'sample_token': sample_annotation.sample_token,
                'instance_token': sample_annotation.instance_token,
                'attribute_tokens': sample_annotation.attribute_tokens,
                'visibility_token': sample_annotation.visibility_token,
                'translation': sample_annotation.translation,
                'size': sample_annotation.size,
                'rotation': sample_annotation.rotation,
                'num_lidar_pts': sample_annotation.num_lidar_pts,
                'num_radar_pts': sample_annotation.num_radar_pts,
                'next': sample_annotation.next,
                'prev': sample_annotation.prev,
            })
        sample_annotations_path = os.path.join(output_path, 'sample_annotations.json')
        with open(sample_annotations_path, 'w') as f:
            json.dump(sample_annotation_json, f, indent=4)


class SampleAnnotation:
    def __init__(self, sample_data, annotation):
        annotation_metadata = annotation.get('metadata', dict()).get('system', dict())
        self.sample_token = sample_data.token
        self.jsons_path = sample_data.jsons_path
        self.items_path = sample_data.items_path
        self.token = "{}_{}".format(annotation.get('id'), sample_data.frame_num)
        self.instance_token = "{}_{}".format(annotation.get('id'), annotation_metadata.get('objectId'))
        self.attribute_tokens = ''
        self.visibility_token = ''
        self.translation = []
        self.size = []
        self.rotation = []
        self.num_lidar_pts = ''
        self.num_radar_pts = 0
        self.next = ''
        self.prev = ''
        self.load_sample_annotation(sample_data=sample_data, annotation=annotation)

    def load_sample_annotation(self, sample_data, annotation):
        """
            load sample annotation parameters to SampleAnnotation object
            :param sample: Sample entity
            :return:
        """
        coordinates = annotation.get('coordinates', dict())
        snapshots = annotation.get('metadata', dict()).get('system', dict()).get('snapshots_')
        for snapshot in snapshots:
            if snapshot.get('frame') <= sample_data.frame_num:
                coordinates = snapshot.get('data', dict())
        self.calc_transformations(coordinates=coordinates)
        self.calc_next(annotation, sample_data.frame_num)
        self.calc_prev(annotation, sample_data.frame_num)
        self.calc_num_lidar_pts(sample_data=sample_data)
        self.calc_attribute_tokens(annotation=annotation)
        self.calc_visibility_token(annotation=annotation)

    def calc_attribute_tokens(self, annotation):
        """
        :param annotation: annotation json
        :return: list of all attribute ids
        """
        self.attribute_tokens = list(
            annotation.get('metadata', dict()).get('system', dict()).get('attributes', dict()).keys())

    def calc_visibility_token(self, annotation):
        """
        value from radio button attribute with id visibility
        :param annotation:
        :return: level of visibility Attribute
        """
        self.visibility_token = annotation.get('metadata', dict()).get('system', dict()).get('attributes', dict()).get(
            'visibility')

    def calc_transformations(self, coordinates):
        """
        :param coordinates: annotation coordinates
        :return: translation, rotation and scale
        """
        position = coordinates.get('position', dict())
        rotation = coordinates.get('rotation', dict())
        scale = coordinates.get('scale', dict())
        self.translation = [position.get('x'), position.get('y'), position.get('z')]
        self.rotation = [rotation.get('x'), rotation.get('y'), rotation.get('z')]
        self.size = [scale.get('x'), scale.get('y'), scale.get('z')]

    @staticmethod
    def get_cube(translation, scale):
        """
        create a cube from annotation
        :param translation: annotation translation
        :param scale: annotation scale
        :return: 2 points on opposite sides of cube diagonal
        """
        x_position = translation[0]
        y_position = translation[1]
        z_position = translation[2]
        x_scale = scale[0]
        y_scale = scale[1]
        z_scale = scale[2]

        points = np.asarray([
            [x_position + x_scale / 2, y_position + y_scale / 2, z_position + z_scale / 2],
            [x_position + x_scale / 2, y_position + y_scale / 2, z_position - z_scale / 2],
            [x_position + x_scale / 2, y_position - y_scale / 2, z_position + z_scale / 2],
            [x_position + x_scale / 2, y_position - y_scale / 2, z_position - z_scale / 2],
            [x_position - x_scale / 2, y_position + y_scale / 2, z_position + z_scale / 2],
            [x_position - x_scale / 2, y_position + y_scale / 2, z_position - z_scale / 2],
            [x_position - x_scale / 2, y_position - y_scale / 2, z_position + z_scale / 2],
            [x_position - x_scale / 2, y_position - y_scale / 2, z_position - z_scale / 2],
        ])
        sorted_points = np.asarray([points[0], points[7]])
        return sorted_points

    def count_points_inside_cube(self, points, translation, scale):
        """
        cube3d  =  numpy array of the shape (8,3) with coordinates in the clockwise order. first the bottom plane is considered then the top one.
        points = array of points with shape (N, 3).

        Returns the indices of the points array which are outside the cube3d
        """
        cube3d = self.get_cube(translation=translation, scale=scale)

        points_inside = list()
        for point in np.asarray(points):
            if min(cube3d[0][0], cube3d[1][0]) < point[0] < max(cube3d[0][0], cube3d[1][0]) and min(cube3d[0][1],
                                                                                                    cube3d[1][1]) < \
                    point[1] < max(cube3d[0][1], cube3d[1][1]) and min(cube3d[0][2], cube3d[1][2]) < point[2] < max(
                cube3d[0][2], cube3d[1][2]):
                points_inside.append(list(point))
        return len(points_inside)

    def get_lidar_metrics(self, pcd_path):
        """
        :param pcd_path: local path to pcd file
        :return:  number of lidar points inside the cube annotation
        """
        pcd = o3d.io.read_point_cloud(pcd_path)
        num_points_in_cube = self.count_points_inside_cube(points=pcd.points, translation=self.translation,
                                                           scale=self.size)
        return num_points_in_cube

    def calc_num_lidar_pts(self, sample_data):
        """
        find path to sampleData (pcd file) and count number of points inside the cube
        :param sample_data: SampleData entity
        :return:
        """
        local_base_name = os.path.basename(sample_data.items_path)
        remote_split = sample_data.file_remote_path.split('/')
        index_common = remote_split.index(local_base_name)
        local_file_path = os.path.join(sample_data.items_path, remote_split[index_common + 1])
        for sub_dir_name in remote_split[index_common + 2:]:
            local_file_path = os.path.join(local_file_path, sub_dir_name)
        self.num_lidar_pts = self.get_lidar_metrics(pcd_path=local_file_path)

    def calc_prev(self, annotation, frame_num):
        """
        sample annotation id on previous frame
        :param annotation:
        :param frame_num:
        :return:
        """
        if frame_num > 0:
            self.prev = "{}_{}".format(annotation.get('id'), frame_num - 1)

    def calc_next(self, annotation, frame_num):
        """
        sample annotation id on next frame
        :param annotation:
        :param frame_num:
        :return:
        """
        annotation_metadata = annotation.get('metadata', dict()).get('system', dict())
        annotation_end_frame = annotation_metadata.get('endFrame', 0)
        if frame_num < annotation_end_frame:
            self.next = "{}_{}".format(annotation.get('id'), frame_num + 1)
