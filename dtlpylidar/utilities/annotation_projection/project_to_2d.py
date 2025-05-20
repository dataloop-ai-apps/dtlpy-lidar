import dtlpy as dl
import os
import numpy as np
import uuid
import json
import dtlpylidar.utilities.transformations as transformations
from tqdm import tqdm


class AnnotationProjection(dl.BaseServiceRunner):
    def __init__(self):
        ...

    @staticmethod
    def get_annotation_metrics(geo):
        """
        Get annotation metrics from geo
        :param geo:
        :return: annotation translation, rotation and scale
        """
        return list(geo[0]), list(geo[1]), list(geo[2])

    @staticmethod
    def get_metrics_snapshot(data):
        """
        Get annotation metrics from snapshot
        :param data: snapshot dictionary
        :return: annotation translation, rotation and scale
        """
        translation = [data.get('position').get('x'), data.get('position').get('y'), data.get('position').get('z')]
        rotation = [data.get('rotation').get('x'), data.get('rotation').get('y'), data.get('rotation').get('z')]
        scale = [data.get('scale').get('x'), data.get('scale').get('y'), data.get('scale').get('z')]
        return translation, scale, rotation

    @staticmethod
    def check_boundaries(width, height, x, y):
        """
        Move point to boundaries if it is outside the image boundaries
        :param width: image width
        :param height: image height
        :param x: point x coordinate
        :param y: point y coordinate
        :return: x, y coordinates of the point after moving it to the boundaries if it was outside the image boundaries
        """
        if x < 0:
            x = 0
        if x > width:
            x = width
        if y < 0:
            y = 0
        if y > height:
            y = height
        return x, y

    def create_annotation(self, label, annotation_pixels, width, height, full_annotations_only):
        """
        Create annotation from 3D cube 8 points projected on 2D image.
        :param label: annotation label
        :param annotation_pixels: annotation 3D cube 8 points projected on 2D image.
        :param width: image width
        :param height: image height
        :param full_annotations_only: if True, only full annotations will be projected to 2D
        :return: cube annotation if at least 2 points are inside the image boundaries.
        """
        # check if at least 2 points are inside the image boundaries
        counter = 0
        if full_annotations_only:
            min_threshold = 8
        else:
            min_threshold = 1
        for annotation_corner in annotation_pixels:
            if (0 < annotation_corner[0] < width) and (0 < annotation_corner[1] < height):
                counter += 1

        # if not enough points are inside the image boundaries skip annotation creation
        if not (counter >= min_threshold):
            return None

        # move points to boundaries if they are outside the image boundaries
        front_tl = self.check_boundaries(width=width,
                                         height=height,
                                         x=annotation_pixels[2][0],
                                         y=annotation_pixels[2][1])
        front_tr = self.check_boundaries(width=width,
                                         height=height,
                                         x=annotation_pixels[3][0],
                                         y=annotation_pixels[3][1])
        front_bl = self.check_boundaries(width=width,
                                         height=height,
                                         x=annotation_pixels[0][0],
                                         y=annotation_pixels[0][1])
        front_br = self.check_boundaries(width=width,
                                         height=height,
                                         x=annotation_pixels[1][0],
                                         y=annotation_pixels[1][1])
        back_tl = self.check_boundaries(width=width,
                                        height=height,
                                        x=annotation_pixels[6][0],
                                        y=annotation_pixels[6][1])
        back_tr = self.check_boundaries(width=width,
                                        height=height,
                                        x=annotation_pixels[7][0],
                                        y=annotation_pixels[7][1])
        back_bl = self.check_boundaries(width=width,
                                        height=height,
                                        x=annotation_pixels[4][0],
                                        y=annotation_pixels[4][1])
        back_br = self.check_boundaries(width=width,
                                        height=height,
                                        x=annotation_pixels[5][0],
                                        y=annotation_pixels[5][1])
        # create cube annotation
        cube = dl.Cube(label=label,
                       front_tl=front_tl,
                       front_tr=front_tr,
                       front_br=front_br,
                       front_bl=front_bl,
                       back_tl=back_tl,
                       back_tr=back_tr,
                       back_br=back_br,
                       back_bl=back_bl)

        # return cube annotation
        return cube

    def calculate_frame_annotations(self, object_id, label, object_visible,
                                    annotation_translation, annotation_rotation, annotation_scale,
                                    lidar_video_content, camera_calibrations, frame_num,
                                    full_annotations_only):
        """
        Calculate frame annotations.
        Iterate over images that correspond with frame and create cube annotation for each image if it is inside the image boundaries.
        :param object_id: annotation object id
        :param label: annotation label
        :param object_visible: original annotation object visibility
        :param annotation_translation: original annotation translation
        :param annotation_rotation: original annotation rotation
        :param annotation_scale: original annotation scale
        :param lidar_video_content: lidar scene video as json
        :param camera_calibrations: camera calibrations
        :param full_annotations_only: if True, only full annotations will be projected to 2D
        :param frame_num:
        :return: None
        """
        # calculate 3D cube points from annotation (PCD normalized)
        points = transformations.calc_cuboid_corners(
            dimensions=annotation_scale
        )
        model_matrix = transformations.calc_transform_matrix(
            rotation=annotation_rotation,
            position=annotation_translation
        )

        # get all images of relevant frame
        frame_images = lidar_video_content.get('frames', list())[frame_num].get('images', list())
        # iterate over images that correspond with frame and create cube annotation for each image if it is inside
        # the image boundaries
        cameras_map = {camera.get('id'): camera for camera in camera_calibrations}

        # iterate over images that correspond with frame
        for idx, image_calibrations in enumerate(frame_images):
            # get image and camera calibrations
            item_id = image_calibrations.get('image_id')
            item = dl.items.get(item_id=item_id)
            camera_id = image_calibrations.get('camera_id')
            camera_calibrations = cameras_map.get(camera_id)
            sensors_data = camera_calibrations.get('sensorsData')

            # calculate view matrix (Default values: Center of the camera is at (0,0,0))
            camera_rotation = sensors_data.get('extrinsic', dict()).get('rotation')
            camera_rotation = [
                camera_rotation.get('x', 0.0),
                camera_rotation.get('y', 0.0),
                camera_rotation.get('z', 0.0),
                camera_rotation.get('w', 1.0)
            ]
            camera_translation = sensors_data.get('extrinsic', dict()).get('position')
            camera_translation = [
                camera_translation.get('x', 0.0),
                camera_translation.get('y', 0.0),
                camera_translation.get('z', 0.0)
            ]
            view_matrix = transformations.calc_transform_matrix(
                rotation=camera_rotation,
                position=camera_translation
            )
            view_matrix = np.linalg.inv(view_matrix)  # inverse of the view matrix (camera to world space)

            # calculate projection matrix (Default values: Orthographic projection)
            intrinsic_data = sensors_data.get('intrinsicData', dict())
            fx = intrinsic_data.get('fx', 1.0)
            fy = intrinsic_data.get('fy', 1.0)
            s = intrinsic_data.get('skew', 0.0)
            cx = intrinsic_data.get('cx', 0.0)
            cy = intrinsic_data.get('cy', 0.0)
            projection_matrix = np.array([
                [fx, s , cx, 0],
                [0 , fy, cy, 0],
                [0 , 0 , 1 , 0],
                [0 , 0 , 0 , 1]
            ])

            mvp = projection_matrix @ view_matrix @ model_matrix
            points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])  # (N, 4)
            projected_points = (mvp @ points_homogeneous.T).T  # (N, 4)
            projected_pixels = projected_points[:, :2] / np.abs(projected_points[:, 2:3])  # (N, 2)

            # create cube annotation if it is inside the image boundaries
            cube_annotation = self.create_annotation(
                label=label,
                annotation_pixels=projected_pixels,
                width=item.width,
                height=item.height,
                full_annotations_only=full_annotations_only
            )
            # if cube annotation is not None add it to the item
            if cube_annotation is None:
                continue
            # create annotation builder
            builder = item.annotations.builder()
            # add annotation to the item
            builder.add(
                annotation_definition=cube_annotation,
                object_id=object_id,
                object_visible=object_visible
            )
            # upload annotation to the item
            item.annotations.upload(builder)

    def project_annotations_to_2d(self, item: dl.Item, full_annotations_only: bool = False):
        """
        Function that projects annotations to 2D from the original lidar scene annotations.
        :param item: DL lidar scene item
        :param full_annotations_only: if True, only full annotations will be projected to 2D
        :return: None
        """
        # download lidar scene video's json
        uid = str(uuid.uuid4())
        items_path = os.path.join(os.getcwd(), uid)
        path = item.download(local_path=items_path)
        with open(path, 'r') as f:
            lidar_video_content = json.load(f)
        # get all annotations of the item
        filters_annotation = dl.Filters(resource=dl.FiltersResource.ANNOTATION)
        filters_annotation.add(field='type', values='cube_3d')
        annotations = item.annotations.list(filters=filters_annotation)
        # get all camera calibrations
        camera_calibrations = lidar_video_content.get('cameras', list())
        # iterate over all annotations
        annotation: dl.Annotation
        for annotation in tqdm(annotations):
            start_frame = annotation.frame_num
            end_frame = annotation.metadata.get('system', dict()).get('endFrame')
            object_id = annotation.object_id

            last_projected_frame = start_frame
            # extract first frame annotation and project it to 2D
            annotation_translation, annotation_scale, annotation_rotation = self.get_annotation_metrics(
                geo=annotation.geo)
            annotation_label = annotation.label
            annotation_object_visible = annotation.object_visible
            self.calculate_frame_annotations(object_id=annotation.object_id,
                                             label=annotation_label,
                                             object_visible=annotation_object_visible,
                                             annotation_translation=annotation_translation,
                                             annotation_rotation=annotation_rotation,
                                             annotation_scale=annotation_scale,
                                             lidar_video_content=lidar_video_content,
                                             camera_calibrations=camera_calibrations,
                                             frame_num=last_projected_frame,
                                             full_annotations_only=full_annotations_only)
            # iterate over all snapshots and project them to 2D
            annotation_snapshots = annotation.metadata.get('system', dict()).get('snapshots_', list())
            for snapshot in annotation_snapshots:
                frame_annotation: dl.entities.FrameAnnotation = dl.entities.FrameAnnotation.from_snapshot(
                    annotation=annotation,
                    _json=snapshot,
                    fps=None
                )
                last_snapshot_frame = frame_annotation.frame_num
                # if there are frames between the last projected frame and the current snapshot frame
                # project them to 2D with the last projected frame metrics
                if last_snapshot_frame != last_projected_frame + 1:
                    for frame_num in range(last_projected_frame + 1, last_snapshot_frame):
                        self.calculate_frame_annotations(object_id=annotation.object_id,
                                                         label=annotation_label,
                                                         object_visible=annotation_object_visible,
                                                         annotation_translation=annotation_translation,
                                                         annotation_rotation=annotation_rotation,
                                                         annotation_scale=annotation_scale,
                                                         lidar_video_content=lidar_video_content,
                                                         camera_calibrations=camera_calibrations,
                                                         frame_num=frame_num,
                                                         full_annotations_only=full_annotations_only)
                # project snapshot to 2D with the snapshot metrics
                annotation_translation, annotation_scale, annotation_rotation = self.get_annotation_metrics(
                    geo=frame_annotation.geo)
                annotation_label = frame_annotation.label
                annotation_object_visible = frame_annotation.object_visible
                self.calculate_frame_annotations(object_id=annotation.object_id,
                                                 label=annotation_label,
                                                 object_visible=annotation_object_visible,
                                                 annotation_translation=annotation_translation,
                                                 annotation_rotation=annotation_rotation,
                                                 annotation_scale=annotation_scale,
                                                 lidar_video_content=lidar_video_content,
                                                 camera_calibrations=camera_calibrations,
                                                 frame_num=last_snapshot_frame,
                                                 full_annotations_only=full_annotations_only)
                last_projected_frame = last_snapshot_frame
            # if there are frames between the last projected frame and the end frame
            # project them to 2D with the last projected frame metrics
            if last_projected_frame < end_frame:
                for frame_num in range(last_projected_frame + 1, end_frame):
                    self.calculate_frame_annotations(object_id=annotation.object_id,
                                                     label=annotation_label,
                                                     object_visible=annotation_object_visible,
                                                     annotation_translation=annotation_translation,
                                                     annotation_rotation=annotation_rotation,
                                                     annotation_scale=annotation_scale,
                                                     lidar_video_content=lidar_video_content,
                                                     camera_calibrations=camera_calibrations,
                                                     frame_num=frame_num,
                                                     full_annotations_only=full_annotations_only)


if __name__ == "__main__":
    dl.setenv('prod')
    item_id = '65670bfcdc18cd144ac914fa'
    full_annotations_only = True

    runner = AnnotationProjection()
    # frames json item ID
    item = dl.items.get(item_id=item_id)
    runner.project_annotations_to_2d(item=item, full_annotations_only=full_annotations_only)
