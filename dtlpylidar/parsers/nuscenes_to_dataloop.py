from dtlpylidar.parser_base import extrinsic_calibrations
from dtlpylidar.parser_base import images_and_pcds, camera_calibrations, lidar_frame, lidar_scene
import os.path
import dtlpy as dl
import pandas as pd
import json
from io import BytesIO
import uuid
import logging

logger = logging.Logger(name='nuscenes_to_dataloop')


class NuscenesToDataloop:

    def __init__(self):
        self.extrinsic_calibrations = dict()
        self.intrinsic_calibrations = dict()
        self.sorted_images = dict()
        self.sorted_pcds = list()

    def extract_extrinsic_calibrations(self, remote_dir, calibrations_csv):
        """
        Extract Extrinsic Calibrations from CSV for all sensors.
        :param remote_dir: Scene name
        :param calibrations_csv: CSV Calibration file as DF.
        :return:
        """
        for row in calibrations_csv.iterrows():
            row_data = row[1]
            if 'velodyne' in row_data.get('frame_id'):
                camera_name = row_data.get('child_frame_id').replace('"', '')
                full_camera_path = os.path.join(remote_dir, camera_name)
                frame_num = int(row_data.get('frameNumber')) - 1
                timestamp = row_data.get('Timestamp')
                if frame_num not in self.extrinsic_calibrations:
                    self.extrinsic_calibrations[frame_num] = dict()
                if full_camera_path not in self.extrinsic_calibrations[frame_num]:
                    self.extrinsic_calibrations[frame_num][full_camera_path] = {
                        'translation': extrinsic_calibrations.Translation(),
                        'rotation': extrinsic_calibrations.QuaternionRotation(),
                        'timestamp': timestamp
                    }
                else:
                    logger.warning("Extrinsic Data for camera: {} frame: {} was found twice in tf.csv file, original "
                                   "data was overwritten".format(full_camera_path, frame_num))

                translation = extrinsic_calibrations.Translation(x=row_data.get('x'),
                                                                 y=row_data.get('y'),
                                                                 z=row_data.get('z'))
                self.extrinsic_calibrations[frame_num][full_camera_path]['translation'] = translation

                rotation = extrinsic_calibrations.QuaternionRotation(x=row_data.get('x.1'),
                                                                     y=row_data.get('y.1'),
                                                                     z=row_data.get('z.1'),
                                                                     w=row_data.get('w'))

                self.extrinsic_calibrations[frame_num][full_camera_path]['rotation'] = rotation

    def extract_intrinsic_calibrations(self, item_name, image_folders):
        """
        Extact Intrinsic Calibrations from json file, and save in dict.
        :param item_name: Scene name.
        :param image_folders: List of paths to local camera calibration json folders.
        :return:
        """
        for image_folder in image_folders:
            intrinsics_path = os.path.join(image_folder, 'camera_info')
            sorted_intrinsic_files = sorted(os.listdir(intrinsics_path),
                                            key=lambda x: int(x.split('_')[1].split('.')[0]))
            for idx, file in enumerate(sorted_intrinsic_files):
                with open(os.path.join(intrinsics_path, file), 'r') as f:
                    data = json.load(f)
                if idx not in self.intrinsic_calibrations:
                    self.intrinsic_calibrations[idx] = dict()
                full_camera_path = image_folder[image_folder.find(item_name):]
                if full_camera_path not in self.intrinsic_calibrations[idx]:
                    self.intrinsic_calibrations[idx][full_camera_path] = {
                        'intrinsic': camera_calibrations.Intrinsic(),
                        'distortion': camera_calibrations.Distortion()
                    }

                    k = eval(data.get('K'))
                    d = eval(data.get('D'))
                    if len(k) >= 7:
                        self.intrinsic_calibrations[idx][full_camera_path]['intrinsic'] = camera_calibrations.Intrinsic(
                            fx=float(k[0]),
                            fy=float(k[5]),
                            cx=float(k[2]),
                            cy=float(k[6]))
                    if len(d) >= 5:
                        self.intrinsic_calibrations[idx][full_camera_path][
                            'distortion'] = camera_calibrations.Distortion(
                            k1=float(d[0]),
                            k2=float(d[1]),
                            k3=float(d[2]),
                            p1=float(d[3]),
                            p2=float(d[4])
                        )

    def sort_images_by_frame(self, image_folders):
        """
        Sort image from each camera folders in a dict by frame number.
        :param image_folders: List of paths to local camera image json folders.
        :return:
        """
        for folder in image_folders:
            images = sorted(os.listdir(os.path.join(folder, 'image_raw')),
                            key=lambda x: int(x.split('_')[1].split('.')[0]))
            for idx, image_name in enumerate(images):
                if idx not in self.sorted_images:
                    self.sorted_images[idx] = list()
                self.sorted_images[idx].append(os.path.join(folder, 'image_raw', image_name))

    def parse_frames_local_sensor(self, dataset, scene_name, upload_scene=True):
        scene = lidar_scene.LidarScene()
        camera_id = 0
        for frame_idx, pcd_filepath in enumerate(self.sorted_pcds):
            lidar_frame_images = list()
            with open(pcd_filepath, 'r') as f:
                pcd_json = json.load(f)
            ground_map_id = pcd_json.get('metadata').get("user", dict()).get('lidar_ground_detection', dict()).get(
                'groundMapId', None)
            dir_name = os.path.join(scene_name, 'velodyne_points')
            pcd_item_rotation = self.extrinsic_calibrations[frame_idx].get(dir_name, dict()).get('rotation')
            pcd_item_translation = self.extrinsic_calibrations[frame_idx].get(dir_name, dict()).get('translation')
            pcd_item_timestamp = self.extrinsic_calibrations[frame_idx].get(dir_name, dict()).get('timestamp')
            scene_pcd_item = images_and_pcds.LidarPcdData(
                item_id=pcd_json.get('id'),
                ground_id=ground_map_id,
                remote_path=pcd_json.get('filename'),
                extrinsic=extrinsic_calibrations.Extrinsic(
                    rotation=pcd_item_rotation,
                    translation=pcd_item_translation
                ),
                timestamp=pcd_item_timestamp
            )

            if frame_idx in self.sorted_images:
                for camera_idx, image_filepath in enumerate(self.sorted_images[frame_idx]):
                    with open(image_filepath, 'r') as f:
                        image_json = json.load(f)
                    dir_name = os.path.normpath(image_json.get('dir'))
                    dir_name = dir_name.split(os.sep)
                    dir_name = os.path.join(*dir_name[:-1])
                    if frame_idx not in self.extrinsic_calibrations or frame_idx not in self.intrinsic_calibrations:
                        continue
                    current_extrinsic_calibrations = self.extrinsic_calibrations[frame_idx]
                    current_intrinsic_calibrations = self.intrinsic_calibrations[frame_idx]
                    camera_translation = current_extrinsic_calibrations.get(dir_name, dict()).get('translation')
                    camera_rotation = current_extrinsic_calibrations.get(dir_name, dict()).get('rotation')
                    camera_timestamp = current_extrinsic_calibrations.get(dir_name, dict()).get('timestamp')
                    intrinsic = current_intrinsic_calibrations.get(dir_name, dict()).get('intrinsic')
                    distortion = current_intrinsic_calibrations.get(dir_name, dict()).get('distortion')

                    lidar_camera = camera_calibrations.LidarCameraData(
                        cam_id=camera_id,
                        intrinsic=intrinsic,
                        extrinsic=extrinsic_calibrations.Extrinsic(
                            rotation=camera_rotation,
                            translation=camera_translation),
                        channel=os.path.basename(dir_name),
                        distortion=distortion
                    )

                    scene.add_camera(lidar_camera)
                    scene_image_item = images_and_pcds.LidarImageData(
                        image_json.get('id'),
                        lidar_camera=lidar_camera,
                        remote_path=image_json.get('filename'),
                        timestamp=camera_timestamp
                    )
                    lidar_frame_images.append(scene_image_item)
                    camera_id += 1

            frame_item = lidar_frame.LidarSceneFrame(
                lidar_frame_pcd=scene_pcd_item,
                lidar_frame_images=lidar_frame_images
            )
            scene.add_frame(frame_item)
        if upload_scene is True:
            buffer = BytesIO()
            buffer.write(json.dumps(scene.to_json(), default=lambda x: None).encode())
            buffer.seek(0)
            buffer.name = 'frames.json'
            dataset.items.upload(remote_path="/{}".format(scene_name), local_path=buffer, overwrite=True,
                                 item_metadata={
                                     'system': {
                                         'shebang': {
                                             'dltype': 'PCDFrames'
                                         }
                                     }
                                 })
        else:
            return scene

    def parse_nuscenes_data(self, scene_folder_item: dl.Item):
        """
        Extract all relevant data from a nuScenes scene, and generates a Dl Lidar Scene
        :param scene_folder_item: Scene Folder item.
        :return:
        """
        if not scene_folder_item.type == 'dir':
            raise Exception("only items of type dir are supported in nuscenes.")
        dataset = scene_folder_item.dataset
        filters = dl.Filters()
        filters.add(field=dl.FiltersKnownFields.DIR, values="{}*".format(scene_folder_item.filename))
        uid = str(uuid.uuid4())
        base_path = "{}_{}".format(dataset.name, uid)
        items_download_path = os.path.join(os.getcwd(), base_path)

        # Download items:
        filters.add(field=dl.FiltersKnownFields.NAME, values='*.zip', operator=dl.FILTERS_OPERATIONS_NOT_EQUAL)
        dataset.items.download(local_path=items_download_path,
                               filters=filters,
                               annotation_options=dl.ViewAnnotationOptions.JSON)

        # Path to items and path to jsons.
        items_path = os.path.join(items_download_path, 'items', scene_folder_item.name)
        jsons_path = os.path.join(items_download_path, 'json', scene_folder_item.name)
        try:
            extrinsic_file_path = os.path.join(items_path, 'tf', 'tf.csv')
            calibrations_csv = pd.read_csv(extrinsic_file_path)
        except (FileNotFoundError, IOError):
            logger.error("tf/tf.csv File is missing, Execution is stopping")
            raise FileNotFoundError

        # List of paths to all items image folders
        image_folders_items = [os.path.join(items_path, folder_name) for folder_name in os.listdir(items_path) if
                               'camera' in folder_name]
        # List of paths to all jsons images folders.
        image_folders_jsons = [os.path.join(jsons_path, folder_name) for folder_name in os.listdir(jsons_path) if
                               'camera' in folder_name]
        # Extract Extrinsic Calibrations from tf.csv file.
        self.extract_extrinsic_calibrations(remote_dir=scene_folder_item.name, calibrations_csv=calibrations_csv)
        # Extract Intrinsic Calibrations from jsons in camera_info Dir for each camera.
        self.extract_intrinsic_calibrations(item_name=scene_folder_item.name, image_folders=image_folders_items)
        # Sort the images by frame from all the scene cameras
        self.sort_images_by_frame(image_folders=image_folders_jsons)
        # Sort PCD Files.
        pcds_path = os.path.join(jsons_path, 'velodyne_points')
        sorted_pcds = sorted(os.listdir(pcds_path),
                             key=lambda x: int(x.split('_')[1].split('.')[0]))
        self.sorted_pcds = [os.path.join(pcds_path, pcd_file) for pcd_file in sorted_pcds]
        # Parse all the data into DL Lidar Scene
        self.parse_frames_local_sensor(dataset=dataset, scene_name=scene_folder_item.name)
