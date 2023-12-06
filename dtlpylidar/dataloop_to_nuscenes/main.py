import dtlpy as dl
import logging
import os
import uuid
import json
from .scenes_jsons import Scenes, Scene, LogFiles, LogFile
from . import values
from .annotations_jsons import Instances, Instance
from multiprocessing.pool import ThreadPool
from time import time
import zipfile
import copy

logger = logging.getLogger(name='nuScenes-converter')


class Runner(dl.BaseServiceRunner):
    def __init__(self):
        self.scenes_information = dict()
        self.log_token = 'scenes_information'
        self.generic_log_used = False
        """
        Init Service attributes here
        :return:
        """

    def single_scene_to_nuscenes(self, item: dl.Item):
        dataset = item.dataset
        if not item.type == 'dir':
            raise Exception("only items of type dir are supported in nuscenes.")
        filters = dl.Filters()
        filters.add(field=dl.FiltersKnownFields.DIR, values="{}*".format(item.filename))
        output_path = self._dataloop_to_nuscenes(dataset=dataset,
                                                 filters=filters)
        self._export_data_as_zip(output_path=output_path, dataset=dataset)

    def dataset_to_nuscenes(self, dataset: dl.Dataset, query=None):
        """
        function that creates a nuScenes schema for all the scenes in a dataset,
         and uploads a zip file including jsons into the dataset.
        :param dataset:
        :param query:
        :return:
        """
        output_path = self._dataloop_to_nuscenes(dataset=dataset,
                                                 query=query)
        self._export_data_as_zip(output_path=output_path, dataset=dataset)

    @staticmethod
    def _export_data_as_zip(output_path, dataset):
        run_time = str(time()).split('.')[0]
        zip_filename = os.path.join(os.getcwd(), '{}_nuscenes_{}.zip'.format(dataset.name, run_time))
        zip_file = zipfile.ZipFile(zip_filename, 'a', zipfile.ZIP_DEFLATED)
        try:
            for root, dirs, files in os.walk(output_path):
                for file in files:
                    filepath = os.path.join(root, file)
                    zip_file.write(filepath, arcname=os.path.relpath(filepath, output_path))
        finally:
            zip_file.close()
        item = dataset.items.upload(local_path=zip_filename, remote_path='/nuScenes_zip')
        return item.id

    def _dataloop_to_nuscenes(self, dataset: dl.Dataset, filters=None, query=None):
        """
        lidar converter from dataloop dataset to nuscenes format.
        :param dataset: dl.Dataset
        :param query: DQL
        :return: path to the output json files.
        """
        if filters is None and query is None:
            filters = dl.Filters()
        if filters is None and query is not None:
            filters = dl.Filters(custom_filter=query)
        uid = str(uuid.uuid4())

        base_path = "{}_{}".format(dataset.name, uid)
        items_path = os.path.join(os.getcwd(), base_path)

        # Download items:
        filters = filters
        filters.add(field=dl.FiltersKnownFields.NAME, values='*.zip', operator=dl.FILTERS_OPERATIONS_NOT_EQUAL)
        dataset.items.download(local_path=items_path,
                               filters=filters,
                               annotation_options=dl.ViewAnnotationOptions.JSON)
        # annotations download
        filters = filters
        filters.add(field=dl.FiltersKnownFields.NAME, values='*.zip', operator=dl.FILTERS_OPERATIONS_NOT_EQUAL)
        annotations_filters = dl.Filters(resource=dl.FiltersResource.ANNOTATION)
        annotations_filters.add(field='type',
                                values='cube_3d')
        path = dataset.download_annotations(local_path=items_path,
                                            filters=filters,
                                            annotation_options=dl.ViewAnnotationOptions.JSON,
                                            annotation_filters=annotations_filters)

        items_path = os.path.join(path, 'items')
        jsons_path = os.path.join(path, 'json')
        try:
            with open(os.path.join(items_path, values.scenes_information), 'r') as f:
                self.scenes_information = json.load(f)
        except (FileNotFoundError, IOError):
            self.scenes_information = {}
            logger.warning("General scenes information file does not exist, checking in scene")
        scenes = Scenes()
        instances = Instances()
        log_files = LogFiles()
        main_pool = ThreadPool(processes=32)
        scenes_dict = copy.copy(items_path)
        for directory in os.listdir(scenes_dict):

            if os.path.isfile(os.path.join(items_path, directory)):
                continue

            try:
                with open(os.path.join(items_path, directory, values.json_video_name), 'r') as f:
                    _ = json.load(f)
            except (FileNotFoundError, IOError):
                logger.warning(
                    "scene: {} does not have a json video of this name: {}".format(directory,
                                                                                   values.json_video_name))
                continue

            try:
                with open(os.path.join(items_path, directory, values.scenes_information), 'r') as f:
                    scenes_information = json.load(f)
                    log_token = "scenes_information_{}".format(directory)
            except (FileNotFoundError, IOError):
                scenes_information = self.scenes_information
                log_token = self.log_token
                logger.warning("scenes information file does not exist, using general if it exists")

            create_nuscenes_inputs = {'scenes': scenes,
                                      'instances': instances,
                                      'log_files': log_files,
                                      'directory': directory,
                                      'jsons_path': jsons_path,
                                      'items_path': items_path,
                                      'scenes_information': scenes_information,
                                      'log_token': log_token}
            main_pool.apply_async(self.create_nuscenes, kwds=create_nuscenes_inputs)

        main_pool.close()
        main_pool.join()
        main_pool.terminate()

        output_path = os.path.join(os.getcwd(), 'output', uid)
        os.makedirs(output_path, exist_ok=True)
        scenes.to_json(output_path=output_path)
        instances.to_json(output_path=output_path)
        log_files.to_json(output_path=output_path)
        category = self.create_category_file(dataset=dataset, scenes_information=scenes_information)
        attributes = self.create_attribute_file(dataset=dataset, scenes_information=scenes_information)
        visibility = self.create_visibility_file(scenes_information=scenes_information)
        with open(os.path.join(output_path, 'category.json'), 'w') as f:
            json.dump(category, f, indent=4)
        with open(os.path.join(output_path, 'attributes.json'), 'w') as f:
            json.dump(attributes, f, indent=4)
        with open(os.path.join(output_path, 'visibility.json'), 'w') as f:
            json.dump(visibility, f, indent=4)
        return output_path

    def create_nuscenes(self, scenes, instances, log_files, directory, jsons_path, items_path, scenes_information,
                        log_token):
        """
        extracts information for the current scene and adds all the information to the dicts that will be exported to json
        :param scenes: dict where all the scenes will be added.
        :param instances: dict where all the instances will be added.
        :param log_files: dict where all the log_files will be added.
        :param directory: current scene name.
        :param jsons_path: path to dl annotation jsons
        :param items_path: path to dl items.
        :param scenes_information: content of the scenes information file
        :param log_token: token for the log object of the current scene.
        :return:
        """

        if not log_token == self.log_token or self.generic_log_used is False:
            log_files.add_log_file(LogFile(scene_information=scenes_information,
                                           token=log_token))
        if log_token == self.log_token:
            self.generic_log_used = True
        instances_list = self.create_instances(jsons_path=os.path.join(jsons_path, directory))
        instances.add_instance(instances=instances_list)
        scene = Scene(dir_name=directory,
                      items_path=os.path.join(items_path, directory),
                      jsons_path=os.path.join(jsons_path, directory),
                      scene_description=scenes_information.get('scene description'),
                      log_token=log_token)
        calibrated_sensors, sensors = scene.create_calibrated_sensors()
        samples = scene.create_samples()
        # add Tables that are connected to scenes.
        scenes.add_scene(dir_name=directory,
                         scene=scene)
        scenes.samples.add_samples(samples=samples)
        scenes.sensors.add_sensor(sensors=sensors)
        scenes.calibrated_sensors.add_calibrated_sensor(calibrated_sensors=calibrated_sensors)

    @staticmethod
    def create_instances(jsons_path):
        """
        finds all annotations of a scene and build the instances json file
        :param jsons_path: path to lidar video annotation file.
        :return:
        """
        lidar_video_annotations_path = os.path.join(jsons_path, values.json_video_name)
        with open(lidar_video_annotations_path, 'r') as f:
            lidar_video_annotations = json.load(f)
        annotations = lidar_video_annotations.get('annotations')
        instances = list()
        for annotation in annotations:
            annotation_metadata = annotation.get('metadata', dict()).get('system', dict())
            annotation_start_frame = annotation_metadata.get('frame', 0)
            annotation_end_frame = annotation_metadata.get('endFrame', 0)
            annotation_object_id = annotation_metadata.get('objectId')
            annotation_label = annotation.get('label')
            first_annotation_token = "{}_{}".format(annotation.get('id'), annotation_start_frame)
            last_annotation_token = "{}_{}".format(annotation.get('id'), annotation_end_frame)
            annotation_id = annotation.get('id')
            instance = Instance(token="{}_{}".format(annotation_id, annotation_object_id),
                                category_token=annotation_label,
                                nbr_annotations=annotation_end_frame - annotation_start_frame,
                                first_annotation_token=first_annotation_token,
                                last_annotation_token=last_annotation_token)
            instances.append(instance)
        return instances

    @staticmethod
    def create_category_file(dataset: dl.Dataset, scenes_information):
        """
        creates the attribute json file from the dataset's recipe
       :return:
       """
        categories_output = list()
        ontology = dataset.ontologies.list()[0]
        for label, idx in ontology.instance_map.items():
            categories_output.append({
                'token': label,
                'name': label,
                'description': scenes_information.get('labels description', dict()).get(
                    label),
                'index': idx
            })
        return categories_output

    @staticmethod
    def create_attribute_file(dataset: dl.Dataset, scenes_information):
        """
        creates the attribute json file from the dataset's recipe
       :return:
       """
        attributes_output = list()
        ontology = dataset.ontologies.list()[0]
        attributes = ontology.metadata.get('attributes', list())
        for attribute in attributes:
            attribute_key = attribute.get('key')
            attributes_output.append({
                'token': attribute_key,
                'name': attribute_key,
                'description': scenes_information.get('attributes description',
                                                      dict()).get(attribute_key)
            })
        return attributes_output

    @staticmethod
    def create_visibility_file(scenes_information):
        """
        extracts the visibility table values from the scene information json file
        :return:
        """
        return scenes_information.get('visibility', dict())
