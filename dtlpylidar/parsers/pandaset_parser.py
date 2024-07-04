from dtlpylidar.parser_base.extrinsic_calibrations import Extrinsic, QuaternionRotation, Translation
from dtlpylidar.parser_base.lidar_frame import LidarSceneFrame
from dtlpylidar.parser_base.lidar_scene import LidarScene
from dtlpylidar.parser_base.images_and_pcds import LidarPcdData, LidarImageData
from dtlpylidar.parser_base.camera_calibrations import LidarCameraData, Intrinsic, Distortion

import os.path
import dtlpy as dl
import json
from io import BytesIO


class PandaSetParser(dl.BaseServiceRunner):
    def __init__(self):
        ...

    @staticmethod
    def parse_frames_local_sensor(dataset, pcd_files, images, folder_name):
        scene = LidarScene()
        camera_id = 0
        for idx, pcd_and_cals in pcd_files.items():
            pcd = pcd_and_cals.get('pcd')
            pcd_extrinsics = pcd_and_cals.get('extrinsics')
            ground_map_id = pcd.metadata.get("user", dict()).get('lidar_ground_detection', dict()).get(
                'groundMapId', None)
            pcd_item = LidarPcdData(item_id=pcd.id,
                                    ground_id=ground_map_id,
                                    extrinsic=pcd_extrinsics,
                                    remote_path=pcd.filename)

            images_lidar = list()
            if idx in images:
                for image_and_cals in images[idx]:
                    image = image_and_cals.get('image')
                    dir_name = '/'.join(image.dir.split('/')[:3])
                    camera = LidarCameraData(cam_id=camera_id,
                                             intrinsic=image_and_cals.get('intrinsics', Intrinsic()),
                                             extrinsic=image_and_cals.get('extrinsics',
                                                                          Extrinsic(rotation=QuaternionRotation(),
                                                                                    translation=Translation())),
                                             distortion=Distortion(),
                                             channel=os.path.basename(dir_name))
                    scene.add_camera(camera)
                    image_lidar = LidarImageData(image.id,
                                                 lidar_camera=camera,
                                                 remote_path=image.filename)
                    images_lidar.append(image_lidar)
                    camera_id += 1

                frame_item = LidarSceneFrame(
                    lidar_frame_pcd=pcd_item,
                    lidar_frame_images=images_lidar,
                )
                scene.add_frame(frame_item)
        buffer = BytesIO()
        buffer.write(json.dumps(scene.to_json(), default=lambda x: None).encode())
        buffer.seek(0)
        buffer.name = 'frames.json'
        item = dataset.items.upload(remote_path="{}".format(folder_name),
                                    local_path=buffer,
                                    overwrite=True,
                                    item_metadata={
                                        'fps': 1,
                                        'system': {
                                            'shebang': {
                                                'dltype': 'PCDFrames'
                                            }
                                        }
                                    })
        return item.id

    @staticmethod
    def extract_calibrations(folder, dataset):
        filters_json = dl.Filters()
        filters_json.add(field="metadata.system.mimetype", values="*json*")
        filters_json.add(field=dl.FiltersKnownFields.DIR, values=folder)
        calibration_files = dataset.items.list(filters=filters_json)
        extrinsics = None
        intrinsics = None
        for item in calibration_files.all():
            if 'poses' in item.name:
                binaries = item.download(save_locally=False)
                extrinsics = json.loads(binaries.getvalue())
            if 'intrinsics' in item.name:
                binaries = item.download(save_locally=False)
                intrinsics = json.loads(binaries.getvalue())
                intrinsics = Intrinsic(fx=intrinsics['fx'],
                                       fy=intrinsics['fy'],
                                       cx=intrinsics['cx'],
                                       cy=intrinsics['cy'],
                                       skew=0.0)
        return extrinsics, intrinsics

    @staticmethod
    def extract_pcd_calibrations(folder, dataset):
        filters_json = dl.Filters()
        filters_json.add(field="metadata.system.mimetype", values="*json")
        filters_json.add(field=dl.FiltersKnownFields.DIR, values="/{}/velodyne_points*".format(folder))
        calibration_files = dataset.items.list(filters=filters_json)
        extrinsics = None
        for item in calibration_files.all():
            if 'poses' in item.name:
                binaries = item.download(save_locally=False)
                extrinsics = json.loads(binaries.getvalue())
        return extrinsics

    def sort_images_by_frame(self, dataset, folder_name):
        image_folders = [f'/{folder_name}/front_camera',
                         f'/{folder_name}/front_left_camera',
                         f'/{folder_name}/left_camera',
                         f'/{folder_name}/back_camera',
                         f'/{folder_name}/right_camera',
                         f'/{folder_name}/front_right_camera']

        sorted_images = dict()
        for folder in image_folders:
            filters_image = dl.Filters()
            filters_image.add(field="metadata.system.mimetype", values="image*")
            filters_image.add(field=dl.FiltersKnownFields.DIR, values=folder)
            images = list(dataset.items.list(filters=filters_image).all())
            images = sorted(images, key=lambda x: int(x.name.split('.jpg')[0]))
            extrinsics, intrinsics = self.extract_calibrations(folder=folder, dataset=dataset)
            for idx, image in enumerate(images):
                if idx not in sorted_images:
                    sorted_images[idx] = list()
                current_extrinsic = extrinsics[idx]
                rotation = QuaternionRotation(x=current_extrinsic.get('heading').get('x'),
                                              y=current_extrinsic.get('heading').get('y'),
                                              z=current_extrinsic.get('heading').get('z'),
                                              w=current_extrinsic.get('heading').get('w'),
                                              )
                translation = Translation(x=current_extrinsic.get('position').get('x'),
                                          y=current_extrinsic.get('position').get('y'),
                                          z=current_extrinsic.get('position').get('z'))
                current_extrinsics = Extrinsic(rotation=rotation, translation=translation)
                sorted_images[idx].append(
                    {
                        'image': image,
                        'intrinsics': intrinsics,
                        'extrinsics': current_extrinsics
                    }
                )
        return sorted_images

    def extract_sort_pcds(self, dataset: dl.Dataset, folder_name):
        pcd_filters = dl.Filters()
        pcd_filters.add(field=dl.FiltersKnownFields.DIR, values=f'/{folder_name}/velodyne_points')
        pcd_filters.add(field=dl.FiltersKnownFields.FILENAME, values='*.pcd')
        pcds = list(dataset.items.list(filters=pcd_filters).all())
        sorted_pcds = sorted(pcds, key=lambda x: int(x.name.split('.pcd')[0]))
        sorted_pcds_dict = dict()
        extrinsics = self.extract_pcd_calibrations(dataset=dataset, folder=folder_name)
        for idx, pcd in enumerate(sorted_pcds):
            current_extrinsic = extrinsics[idx]
            rotation = QuaternionRotation(x=current_extrinsic.get('heading').get('x'),
                                          y=current_extrinsic.get('heading').get('y'),
                                          z=current_extrinsic.get('heading').get('z'),
                                          w=current_extrinsic.get('heading').get('w'),
                                          )
            translation = Translation(x=current_extrinsic.get('position').get('x'),
                                      y=current_extrinsic.get('position').get('y'),
                                      z=current_extrinsic.get('position').get('z'))
            current_extrinsics = Extrinsic(rotation=rotation, translation=translation)
            sorted_pcds_dict[idx] = {'pcd': pcd,
                                     'extrinsics': current_extrinsics}
        return sorted_pcds_dict

    def create_json_calibration_files(self, scene_dir: dl.Item):
        if not scene_dir.type == 'dir':
            raise Exception('Item is not a directory')
        dataset = scene_dir.dataset
        sorted_pcds = self.extract_sort_pcds(folder_name=scene_dir.name, dataset=dataset)
        sorted_images = self.sort_images_by_frame(folder_name=scene_dir.name, dataset=dataset)
        item_id = self.parse_frames_local_sensor(dataset=dataset,
                                                 pcd_files=sorted_pcds,
                                                 folder_name=scene_dir.name,
                                                 images=sorted_images)
        return item_id
