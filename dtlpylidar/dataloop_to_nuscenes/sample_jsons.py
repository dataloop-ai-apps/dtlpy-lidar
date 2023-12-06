import json
import os.path
from . import values
from .annotations_jsons import SampleAnnotation, SampleAnnotations
from multiprocessing.pool import ThreadPool


class LidarSegmentations:
    def __init__(self):
        self.lidar_seg_output = dict()

    def add_lidar_seg(self, lidar_seg):
        self.lidar_seg_output[lidar_seg.token] = lidar_seg

    def to_json(self, output_path):
        lidar_seg_json = list()
        for lidar_seg_token, lidar_seg in self.lidar_seg_output.items():
            lidar_seg_json.append({
                'token': lidar_seg_token,
                'filename': lidar_seg.filename,
                'sample_data_token': lidar_seg.sample_data_token
            })
        lidar_seg_path = os.path.join(output_path, 'lidarseg.json')
        with open(lidar_seg_path, 'w') as f:
            json.dump(lidar_seg_json, f, indent=4)


class LidarSeg:
    def __init__(self, pcd_filename, sample_data_token):
        self.token = pcd_filename
        self.filename = pcd_filename
        self.sample_data_token = sample_data_token


class Sensors:
    def __init__(self):
        self.sensors_output = dict()

    def add_sensor(self, sensors):
        """
        add all sensors to the dict that will be the final sensors.json output
        :param sensors: list of sensors
        :return:
        """
        for sensor in sensors:
            self.sensors_output[sensor.token] = sensor

    def to_json(self, output_path):
        sensor_json = list()
        for sensor_token, sensor in self.sensors_output.items():
            sensor_json.append({
                'token': sensor_token,
                'channel': sensor.chanel,
                'modality': sensor.modality,
            })
        sensor_output_path = os.path.join(output_path, 'sensor.json')
        with open(sensor_output_path, 'w') as f:
            json.dump(sensor_json, f, indent=4)


class Sensor:
    def __init__(self, token, channel, modality):
        self.token = token
        self.chanel = channel
        self.modality = modality


class CalibratedSensor:
    def __init__(self, calibrations: dict, sensortype, scene_token, items_path, jsons_path, sensor_token):
        """
        calibrated sensor entity
        :param calibrations: lidar video json conent
        :param sensortype:'camera' / 'lidar'
        :param scene_token: folder name
        :param items_path: local path to downloaded dataloop items
        :param jsons_path: local path to downloaded dataloop jsons
        :param sensor_token: sensor token from Sensor entity
        """
        self.calibrations = calibrations
        self.sensortype = sensortype
        self.items_path = items_path
        self.jsons_path = jsons_path
        self.scene_token = scene_token
        self.token = ''
        self.sensor_token = sensor_token
        self.translation = []
        self.rotation = []
        self.camera_intrinsic = []
        self.load_calibrated_sensor()

    def load_calibrated_sensor(self):
        """
        find transformation and rotation for each sensor
        6 cameras per scene
        1 lidar sensor per scene
        calibrations taken from the lidar video json conent
        only camera sensors have intrisics
        :return:
        """
        if self.sensortype == 'camera':
            self.get_transformation_camera(transformation_type='position')
            self.get_transformation_camera(transformation_type='rotation')
            self.get_intrinsics_camera()
        else:
            self.get_transformations_lidar(transformation_type='translation')
            self.get_transformations_lidar(transformation_type='rotation')
        self.set_token()

    def set_token(self):
        """
        {scene_token}_{camera_id} for cameras
        {scene_token}_{lidar} for lidar sensor
        :return:
        """
        if self.sensortype == 'camera':
            sensor_id = self.calibrations.get('id')
        else:
            sensor_id = 'lidar'
        self.token = '{}_{}'.format(self.scene_token, sensor_id)

    def get_intrinsics_camera(self):
        """
        get camera intrinsic matrix from frame.json file
        :return:
        """
        self.camera_intrinsic = self.calibrations.get('sensorsData', dict()).get('intrinsicMatrix', list)

    def get_transformation_camera(self, transformation_type):
        """
        get camera extrinsics (translation and rotation) from frames.json file
        :param transformation_type:
        :return:
        """
        if transformation_type not in ['position', 'rotation']:
            raise Exception('transformation must be rotation or translation')
        transformation = self.calibrations.get('sensorsData', dict()).get('extrinsic', dict()).get(transformation_type,
                                                                                                   dict())
        transformation = [coordinate for coordinate in transformation.values()]
        if 'position' == transformation_type:
            self.translation = transformation
        else:
            self.rotation = transformation

    def get_transformations_lidar(self, transformation_type):
        """
        get lidar sensor extrinsics (translation and rotation) from frames.json file
        :param transformation_type:
        :return:
        """
        if transformation_type not in ['translation', 'rotation']:
            raise Exception('transformation must be rotation or translation')
        transformation = self.calibrations.get(transformation_type, dict())
        transformation = [coordinate for coordinate in transformation.values()]
        if 'translation' == transformation_type:
            self.translation = transformation
        else:
            self.rotation = transformation


class CalibratedSensors:
    def __init__(self):
        self.calibrated_sensors_output = dict()

    def add_calibrated_sensor(self, calibrated_sensors: list):
        """
        add all calibrated sensors to the dict that will be the final calibrated_sensors.json output
        :param calibrated_sensors: list of calibrated sensors
        :return:
        """
        for calibrated_sensor in calibrated_sensors:
            self.calibrated_sensors_output[calibrated_sensor.token] = calibrated_sensor

    def to_json(self, output_path):
        calibrated_sensor_json = list()
        for calibrated_sensor_token, calibrated_sensor in self.calibrated_sensors_output.items():
            calibrated_sensor_json.append({
                'token': calibrated_sensor_token,
                'sensor_token': calibrated_sensor.sensor_token,
                'translation': calibrated_sensor.translation,
                'rotation': calibrated_sensor.rotation,
                'camera_intrinsic': calibrated_sensor.camera_intrinsic
            })
        calibrated_sensor_output_path = os.path.join(output_path, 'calibrated_sensor.json')
        with open(calibrated_sensor_output_path, 'w') as f:
            json.dump(calibrated_sensor_json, f, indent=4)


class Sample:
    def __init__(self, scene_token, items_path, jsons_path, lidar_frames, frame_num, sensor, cam_idx=None):
        """
        initiate a sample object, each sample describes a file in the scene
        :param scene_token: scene name
        :param items_path: local path to downloaded dataloop items
        :param jsons_path: local path to downloaded dataloop jsons
        :param lidar_frames: lidar video json content
        :param frame_num: frame number
        :param sensor: 'lidar' / 'camera'
        :param cam_idx: image number
        """
        self.items_path = items_path
        self.jsons_path = jsons_path
        self.scene_token = scene_token
        self.sensor = sensor
        self.token = ''
        self.timestamp = ''
        self.next = ''
        self.prev = ''
        self.frame_num = frame_num
        self.cam_idx = cam_idx
        if cam_idx is None:
            self.load_sample_pcd(lidar_frames=lidar_frames, frame_num=frame_num)
        else:
            self.load_sample_image(lidar_frames=lidar_frames, frame_num=frame_num, cam_idx=cam_idx)

    def load_sample_pcd(self, lidar_frames, frame_num):
        """
        load pcd sample data from lidar video json content
        :param lidar_frames:
        :param frame_num:
        :return:
        """
        self.token = self.find_pcd_item_id(lidar_frames, frame_num)
        self.timestamp = lidar_frames[frame_num].get('lidar', dict()).get('timestamp', None)
        if frame_num > 0:
            self.prev = self.find_pcd_item_id(lidar_frames, frame_num - 1)
        if frame_num < len(lidar_frames) - 1:
            self.next = self.find_pcd_item_id(lidar_frames, frame_num + 1)

    def load_sample_image(self, lidar_frames, frame_num, cam_idx):

        """
        load image sample data from lidar video json content
        :param lidar_frames:
        :param frame_num:
        :param cam_idx:
        :return:
        """

        self.token = self.find_image_item_id(lidar_frames, frame_num, cam_idx)
        self.timestamp = lidar_frames[frame_num].get('images', list)[cam_idx].get('timestamp', None)
        if frame_num > 0:
            self.prev = self.find_image_item_id(lidar_frames, frame_num - 1, cam_idx)
        if frame_num < len(lidar_frames) - 1:
            self.next = self.find_image_item_id(lidar_frames, frame_num + 1, cam_idx)

    @staticmethod
    def find_pcd_item_id(frames_list, frame_num):
        return frames_list[frame_num].get('lidar', dict()).get('lidar_pcd_id')

    @staticmethod
    def find_image_item_id(frames_list, frame_num, cam_idx):
        return frames_list[frame_num].get('images', list)[cam_idx].get('image_id')

    def create_ego_pose(self):
        """
        create ego pose entity
        :return: EgoPose
        """
        return EgosPose(sample=self)

    def create_sample_data(self):
        """
        create sample data entity
        :return: SampleData
        """
        sample_data = SampleData(sample=self)
        return sample_data


class Samples:
    def __init__(self):
        self.samples_output = dict()
        self.ego_poses = EgosPoses()
        self.samples_data = SamplesData()

    def add_samples(self, samples):
        """
        add samples to final samples dict (output)
        for each sample, create a sample_data object for sample_data.json output file
        for pcd samples, create an "ego_pose" object for "ego_pose.json output file
        :param samples: list of samples
        :return:
        """
        for sample in samples:
            self.samples_output[sample.token] = sample
            if sample.sensor == 'lidar':
                self.ego_poses.add_ego_pose(sample.create_ego_pose())
            sample_data = sample.create_sample_data()
            self.samples_data.add_sample_data(sample_data)

    def to_json(self, output_path):
        self.ego_poses.to_json(output_path=output_path)
        self.samples_data.to_json(output_path=output_path)
        samples_json = list()
        for sample_token, sample in self.samples_output.items():
            samples_json.append({
                'token': sample_token,
                'timestamp': sample.timestamp,
                'scene_token': sample.scene_token,
                'next': sample.next,
                'prev': sample.prev,

            })
        sample_output_path = os.path.join(output_path, 'sample.json')
        with open(sample_output_path, 'w') as f:
            json.dump(samples_json, f, indent=4)


class EgosPose:
    def __init__(self, sample: Sample):
        """
        ego pose object, describing the relative position of the ego in each pcd sample
        :param sample: sample entity
        """
        self.token = "{}_{}".format(sample.scene_token, sample.token)
        self.translation = []
        self.rotation = []
        self.timestamp = sample.timestamp
        self.load_ego_pose(sample=sample)

    def load_ego_pose(self, sample):
        """
        for each frame in the lidar video json conent find the rotation and translation of the ego
        (same values as the lidar sensor calibrations)
        :param sample:
        :return:
        """
        lidar_video_path = os.path.join(sample.items_path, values.json_video_name)
        with open(lidar_video_path, 'r') as f:
            lidar_data = json.load(f)
        lidar_frame = lidar_data.get('frames', list())[sample.frame_num]

        translation = lidar_frame.get('translation')
        self.translation = [translation.get('x'), translation.get('y'), translation.get('z')]
        rotation = lidar_frame.get('rotation')
        self.rotation = [rotation.get('x'), rotation.get('y'), rotation.get('z'), rotation.get('w')]


class EgosPoses:
    def __init__(self):
        self.ego_pose_output = dict()

    def add_ego_pose(self, ego_pose: EgosPose):
        """
        add single ego pose to ego_poses dict
        :param ego_pose:
        :return:
        """
        self.ego_pose_output[ego_pose.token] = ego_pose

    def to_json(self, output_path):
        ego_pose_json = list()
        for ego_pose_token, ego_pose in self.ego_pose_output.items():
            ego_pose_json.append({
                'token': ego_pose_token,
                'translation': ego_pose.translation,
                'rotation': ego_pose.rotation,
                'timestamp': ego_pose.timestamp
            })
        ego_pose_path = os.path.join(output_path, 'ego_pose.json')
        with open(ego_pose_path, 'w') as f:
            json.dump(ego_pose_json, f, indent=4)


class SampleData:
    def __init__(self, sample: Sample):
        """
        :param sample: Sample entity
        """
        self.jsons_path = sample.jsons_path
        self.items_path = sample.items_path
        self.token = "{}_{}".format(sample.token, sample.frame_num)
        self.sample_token = sample.token
        self.timestamp = sample.timestamp
        self.ego_pose_token = "{}_{}".format(sample.scene_token, sample.token)
        self.next = ''
        self.prev = ''
        self.frame_num = sample.frame_num
        self.calibrated_sensor_token = ''
        self.filename = ''
        self.fileformat = ''
        self.width = 0
        self.height = 0
        self.is_key_frame = False
        self.file_remote_path = ''
        self.load_sample_data(sample=sample)

    def load_sample_data(self, sample: Sample):
        """
        load sample data parameters to SampleData object
        :param sample: Sample entity
        :return:
        """
        # calculate next and prev:
        if not sample.next == '':
            self.next = "{}_{}".format(sample.next, self.frame_num + 1)
        if not sample.prev == '':
            self.prev = "{}_{}".format(sample.prev, self.frame_num - 1)
        # Key frames are all 2Hz raw files (The entire scene is 10Hz)
        if self.frame_num % 5 == 0:
            self.is_key_frame = True
        # find the json file and locate the frame number of the sample
        lidar_video_path = os.path.join(sample.items_path, values.json_video_name)
        with open(lidar_video_path, 'r') as f:
            lidar_data = json.load(f)
        frame_calibrations = lidar_data.get('frames', list)[sample.frame_num]
        if sample.sensor == 'lidar':
            sensor_id = 'lidar'
            self.calc_name_format(frame_calibrations.get('lidar', dict()))
        else:
            image_calibrations = frame_calibrations.get('images', list)[sample.cam_idx]
            sensor_id = image_calibrations.get('camera_id', "N/A")
            if not sensor_id == "N/A":
                sensor_id = int(sensor_id) % 6
            self.calc_name_format(image_calibrations)
            self.get_dimensions(sample.jsons_path)
        self.calibrated_sensor_token = '{}_{}'.format(sample.scene_token, sensor_id)

    def get_dimensions(self, jsons_path):
        """
        function that gets the width and height (for images only) from the image's json file.
        :param jsons_path: path to dataset json files
        :return:
        """
        local_base_name = os.path.basename(jsons_path)
        remote_split = self.file_remote_path.split('/')
        index_common = remote_split.index(local_base_name)
        local_file_path = os.path.join(jsons_path, remote_split[index_common + 1])
        for sub_dir_name in remote_split[index_common + 2:]:
            local_file_path = os.path.join(local_file_path, sub_dir_name)
        base_name = os.path.basename(local_file_path)
        json_base_name = base_name.split('.')[0] + '.json'
        local_file_path = local_file_path.replace(base_name, json_base_name)
        with open(local_file_path, 'r') as f:
            data = json.load(f)
        self.height = data.get('metadata', dict()).get('system', dict()).get('height', "N/A")
        self.width = data.get('metadata', dict()).get('system', dict()).get('width', "N/A")

    def calc_name_format(self, calibrations):
        """
        from the frames.json file, get the file name and file extension
        :param calibrations:
        :return:
        """
        file_remote_path = calibrations.get('remote_path', '')
        file_name = os.path.basename(file_remote_path)
        if '.' in file_name:
            self.filename = file_remote_path
            self.fileformat = file_name.split('.')[-1]
        self.file_remote_path = file_remote_path

    def create_lidar_seg(self):
        pcd_filename = "{}.{}".format(self.filename, self.fileformat)
        lidar_seg = LidarSeg(pcd_filename=pcd_filename, sample_data_token=self.token)
        return lidar_seg

    def create_sample_annotations(self):
        """
        get all annotations on file
        :return: list of SampleAnnotation entity
        """
        sample_annotations = list()
        lidar_video_annotations_path = os.path.join(self.jsons_path, values.json_video_name)
        with open(lidar_video_annotations_path, 'r') as f:
            lidar_video_annotations = json.load(f)
        annotations = lidar_video_annotations.get('annotations', list)
        annotation_pool = ThreadPool(processes=32)
        for annotation in annotations:
            create_annotation_inputs = {'annotation': annotation,
                                        'sample_annotations': sample_annotations}
            annotation_pool.apply_async(self.create_annotations, kwds=create_annotation_inputs)
        annotation_pool.close()
        annotation_pool.join()
        annotation_pool.terminate()
        return sample_annotations

    def create_annotations(self, annotation, sample_annotations):
        annotation_metadata = annotation.get('metadata', dict()).get('system', dict())
        annotation_start_frame = annotation_metadata.get('frame', 0)
        annotation_end_frame = annotation_metadata.get('endFrame', 0)
        if annotation_start_frame <= self.frame_num <= annotation_end_frame:
            sample_annotation = SampleAnnotation(sample_data=self, annotation=annotation)
            sample_annotations.append(sample_annotation)


class SamplesData:
    def __init__(self):
        self.samples_data_output = dict()
        self.sample_annotations = SampleAnnotations()
        self.lidar_segmentations = LidarSegmentations()

    def add_sample_data(self, sample_data: SampleData):
        """
        add a single sample data to samples_data dict
        if sample_data is a pcd file, find all annotations for the specific frame.
        :param sample_data: SampleData entity
        :return:
        """
        self.samples_data_output[sample_data.token] = sample_data
        lidar_seg = sample_data.create_lidar_seg()
        self.lidar_segmentations.add_lidar_seg(lidar_seg=lidar_seg)
        if 'pcd' in sample_data.fileformat:
            sample_annotations = sample_data.create_sample_annotations()
            self.sample_annotations.add_sample_annotation(sample_annotations)

    def to_json(self, output_path):
        self.sample_annotations.to_json(output_path=output_path)
        self.lidar_segmentations.to_json(output_path=output_path)
        sample_data_json = list()
        for sample_data_token, sample_data in self.samples_data_output.items():
            sample_data_json.append({
                'token': sample_data_token,
                'sample_token': sample_data.sample_token,
                'ego_pose_token': sample_data.ego_pose_token,
                'calibrated_sensor_token': sample_data.calibrated_sensor_token,
                'filename': sample_data.filename,
                'fileformat': sample_data.fileformat,
                'width': sample_data.width,
                'height': sample_data.height,
                'timestamp': sample_data.timestamp,
                'is_key_frame': sample_data.is_key_frame,
                'next': sample_data.next,
                'prev': sample_data.prev
            })
        sample_data_path = os.path.join(output_path, 'sample_data.json')
        with open(sample_data_path, 'w') as f:
            json.dump(sample_data_json, f, indent=4)
