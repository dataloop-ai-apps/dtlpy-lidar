import logging
import os
import json
from .sample_jsons import CalibratedSensors, CalibratedSensor, Sensor, Sensors, \
    Sample, Samples
from . import values

logger = logging.getLogger(name='nuScenes-converter')


class LogFiles:
    def __init__(self):
        self.log_file_output = list()

    def add_log_file(self, log_file):
        self.log_file_output.append(log_file.log_file_data)

    def to_json(self, output_path):
        scene_output_path = os.path.join(output_path, 'log.json')
        with open(scene_output_path, 'w') as f:
            json.dump(self.log_file_output, f, indent=4)


class LogFile:
    def __init__(self, token, scene_information):
        """
        log entity.
        :param token: entity token
        :param scene_information: scene information json file content.
        """
        self.log_file_data = {
            'token': token,
            'logfile': scene_information.get('log', dict()).get('logfile', None),
            'vehicle': scene_information.get('log', dict()).get('vehicle', None),
            'date_captured': scene_information.get('log', dict()).get('date_captured', None),
            'location': scene_information.get('log', dict()).get('location', None)

        }


class Scene:
    def __init__(self, dir_name, items_path, jsons_path, scene_description, log_token):
        """
        Scene entity, each scene describes a folder inside a dataset
        :param dir_name: folder name
        :param items_path: local path to downloaded dataloop items
        :param jsons_path: local path to downloaded dataloop jsons
        :param scene_description: scene description from scenes_information.json
        """
        self.token = dir_name
        self.items_path = items_path
        self.jsons_path = jsons_path
        self.description = scene_description
        self.log_token = log_token
        self.nbr_samples = ''
        self.first_sample_token = ''
        self.last_sample_token = ''
        self.sorted_frames = None
        self.load_scene()

    def load_scene(self):
        """
        find the first frame and last frame id for the pcds in the scene.
        :return:
        """
        self.first_sample_token = self.get_frame_id(idx=0)
        self.last_sample_token = self.get_frame_id(idx=-1)
        self.nbr_samples = len(os.listdir(os.path.join(self.items_path, values.pcd_dir_name)))

    def get_frame_id(self, idx):
        """
        in nuscenes format pcds are located in velodyne_points, navigate to the local directory and get the item ids.
        :param idx: frame number (function called by __init__ twice, once for first frame once for last frame)
        :return: dataloop frame_id
        """
        pcds_json_path = os.path.join(self.jsons_path, values.pcd_dir_name)
        frames = os.listdir(pcds_json_path)
        if len(frames) > 0:
            sorted_frames = sorted(frames, key=lambda x: int(x.split('.')[0].split('_')[1]))
            self.sorted_frames = sorted_frames
            with open(os.path.join(pcds_json_path, sorted_frames[idx]), 'r') as f:
                frame_id = json.load(f).get('id', None)
        return frame_id

    def create_calibrated_sensors(self):
        """
        Create a calibrated sensor and sensor for each one of the 6 cameras in frame json and the lidar sensor
        finds the lidar video json which includes all this data sensors calibrations in it.
        :return: list of sensors and calibrated sensors
        """
        logger.info("creating calibrated sensors and sensors")
        calibrated_sensors = list()
        sensors = list()
        lidar_video_path = os.path.join(self.items_path, values.json_video_name)
        with open(lidar_video_path, 'r') as f:
            lidar_data = json.load(f)
        lidar_frames = lidar_data.get('frames', list())
        if len(lidar_frames) > 0:
            single_lidar_frame = lidar_frames[0]
        else:
            raise Exception('Empty lidar sequence, please review the data again')
        cameras = lidar_data.get('cameras', list())
        logger.info('found {} cameras'.format(len(cameras)))
        for idx, camera in enumerate(cameras):
            if idx == 6:
                break
            calibrated_sensor, sensor = self.create_single_calibrated_sensor(calibrations=camera,
                                                                             sensortype='camera',
                                                                             partial_sensor_token=idx)
            calibrated_sensors.append(calibrated_sensor)
            sensors.append(sensor)
        calibrated_sensor, sensor = self.create_single_calibrated_sensor(calibrations=single_lidar_frame,
                                                                         sensortype='lidar',
                                                                         partial_sensor_token='lidar')
        calibrated_sensors.append(calibrated_sensor)
        sensors.append(sensor)
        return calibrated_sensors, sensors

    def create_single_calibrated_sensor(self, calibrations, sensortype, partial_sensor_token):
        """
        create a single sensor, and calibrated sensor
        :param calibrations: single object from the lidar video json
        :param sensortype:camera / lidar
        :param partial_sensor_token: either "lidar" or {camera_id}
        :return:
        """
        if sensortype == 'camera':
            sensor = Sensor(token="{}_{}".format(partial_sensor_token, self.token),
                            channel=calibrations.get('channel', 'N/A'),
                            modality='camera')
        else:
            sensor = Sensor(token='lidar_{}'.format(self.token), channel='LIDAR_TOP', modality='lidar')
        calibrated_sensor = CalibratedSensor(calibrations=calibrations,
                                             sensortype=sensortype,
                                             items_path=self.items_path,
                                             jsons_path=self.jsons_path,
                                             scene_token=self.token,
                                             sensor_token=sensor.token)
        return calibrated_sensor, sensor

    def create_samples(self):
        """
        each sample is a pcd or an image, for each file in the scene, create a sample entity
        all the relevant file information can be found in the lidar video json, so iterate over all frames
        each frame has data about a single pcd and the 6 cameras.
        :return:
        """
        samples = list()
        lidar_video_path = os.path.join(self.items_path, values.json_video_name)
        with open(lidar_video_path, 'r') as f:
            lidar_data = json.load(f)
        lidar_frames = lidar_data.get('frames', list())
        logger.info('found {} frames'.format(len(lidar_frames)))
        for idx, lidar_frame in enumerate(lidar_frames):
            samples.append(Sample(scene_token=self.token,
                                  items_path=self.items_path,
                                  jsons_path=self.jsons_path,
                                  lidar_frames=lidar_frames,
                                  frame_num=idx,
                                  sensor='lidar'))
            for idx2, image in enumerate(lidar_frame.get('images', list)):
                logger.info('found {} images for frame {}'.format(lidar_frame.get('images', list), idx))
                samples.append(Sample(scene_token=self.token,
                                      items_path=self.items_path,
                                      jsons_path=self.jsons_path,
                                      lidar_frames=lidar_frames,
                                      frame_num=idx,
                                      sensor='camera',
                                      cam_idx=idx2))
        return samples


class Scenes:
    def __init__(self):
        self.scenes_output = dict()
        self.calibrated_sensors = CalibratedSensors()
        self.sensors = Sensors()
        self.samples = Samples()

    def add_scene(self, dir_name, scene: Scene):
        """
        add a single scene to scenes output dict
        :param dir_name:
        :param scene:
        :return:
        """
        self.scenes_output[scene.token] = scene

    def to_json(self, output_path):
        self.calibrated_sensors.to_json(output_path=output_path)
        self.sensors.to_json(output_path=output_path)
        self.samples.to_json(output_path=output_path)
        scenes_json = list()
        for scene_token, scene in self.scenes_output.items():
            scenes_json.append({
                'token': scene_token,
                'name': scene.token,
                'description': scene.description,
                'log_token': scene.log_token,
                'nbr_samples': scene.nbr_samples,
                'first_sample_token': scene.first_sample_token,
                'last_sample_token': scene.last_sample_token
            })
        scene_output_path = os.path.join(output_path, 'scene.json')
        with open(scene_output_path, 'w') as f:
            json.dump(scenes_json, f, indent=4)
