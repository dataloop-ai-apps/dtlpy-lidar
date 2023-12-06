import dtlpy as dl
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import uuid
import json
import dtlpylidar.utilities.transformations as transformations


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
    def apply_camera_projection(points, camera_calibrations):
        """
        Apply camera projection on 3D points
        :param points: 3D cube 8 points
        :param camera_calibrations: camera calibration dictionary
        :return: 3D cube 8 points projected on 2D image.
        """
        sensors_data = camera_calibrations.get('sensorsData')
        # intrinsic calibrations
        fx = sensors_data.get('intrinsicData', dict()).get('fx')
        fy = sensors_data.get('intrinsicData', dict()).get('fy')
        s = sensors_data.get('intrinsicData', dict()).get('skew')
        cx = sensors_data.get('intrinsicData', dict()).get('cx')
        cy = sensors_data.get('intrinsicData', dict()).get('cy')
        # intrinsic Matrix
        intrinsic = np.array([[fx, s, cx],
                              [0, fy, cy],
                              [0, 0, 1]])

        # Move the cube to the camera position
        points += np.array([sensors_data.get('extrinsic', dict()).get('position').get('x'),
                            sensors_data.get('extrinsic', dict()).get('position').get('y'),
                            sensors_data.get('extrinsic', dict()).get('position').get('z')])
        # Rotate the cube to the camera rotation
        r_box = R.from_quat([sensors_data.get('extrinsic', dict()).get('rotation').get('x'),
                             sensors_data.get('extrinsic', dict()).get('rotation').get('y'),
                             sensors_data.get('extrinsic', dict()).get('rotation').get('z'),
                             sensors_data.get('extrinsic', dict()).get('rotation').get('w')])

        points = np.dot(np.linalg.inv(r_box.as_matrix()), points.transpose()).transpose()

        # Apply camera projection
        image_pts = np.dot(intrinsic, points.transpose()).transpose()
        # Normalize image points to 2D
        norm_image_pts = [pt[:2] / np.abs(pt[2]) for pt in image_pts]
        # return 3D cube 8 points projected on 2D image.
        return norm_image_pts

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

    def create_annotation(self, annotation_pixels, annotation, width, height):
        """
        Create annotation from 3D cube 8 points projected on 2D image.
        :param annotation_pixels: annotation 3D cube 8 points projected on 2D image.
        :param annotation:  original annotation
        :param width: image width
        :param height: image height
        :return: cube annotation if at least 5 points are inside the image boundaries.
        """
        counter = 0
        # check if at least 5 points are inside the image boundaries
        for annotation_corner in annotation_pixels:
            if annotation_corner[0] < 0 or annotation_corner[0] > height:
                counter += 1
                continue
            if annotation_corner[1] < 0 or annotation_corner[1] > width:
                counter += 1
                continue
        # if at least 5 points are inside the image boundaries create annotation
        if counter >= 5:
            return
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
        cube = dl.Cube(label=annotation.label,
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

    def iterate_image(self, cameras, images, points, annotation):
        """
        Iterate over images that correspond with frame and create cube annotation for each image if it is inside the image boundaries.
        :param cameras: all cameras calibrations of relevant frame
        :param images: all images of relevant frame
        :param points: 3D cube 8 points
        :param annotation: original annotation
        :return: None
        """
        # iterate over images that correspond with frame
        for idx, image_calibrations in enumerate(images):
            # get image and camera calibrations
            item_id = image_calibrations.get('image_id')
            item = dl.items.get(item_id=item_id)
            camera_id = image_calibrations.get('camera_id')
            camera_calibrations = cameras[int(camera_id)]
            # apply camera projection
            projection = self.apply_camera_projection(points=np.copy(points),
                                                      camera_calibrations=camera_calibrations)
            # create cube annotation if it is inside the image boundaries
            cube_annotation = self.create_annotation(annotation_pixels=projection, annotation=annotation,
                                                     width=item.width,
                                                     height=item.height)
            # if cube annotation is not None add it to the item
            if cube_annotation is None:
                continue
            # create annotation builder
            builder = item.annotations.builder()
            # add annotation to the item
            builder.add(annotation_definition=cube_annotation,
                        object_id=annotation.object_id)
            # upload annotation to the item
            item.annotations.upload(builder)

    def calculate_frame_annotations(self, annotation, annotation_translation, annotation_rotation, annotation_scale,
                                    lidar_video_content, camera_calibrations, frame_num):
        """
        Calculate frame annotations.
        :param annotation: original annotaiton
        :param annotation_translation: original annotation translation
        :param annotation_rotation: original annotation rotation
        :param annotation_scale: original annotation scale
        :param lidar_video_content: lidar scene video as json
        :param camera_calibrations: camera calibrations
        :param frame_num:
        :return:
        """
        # calculate 3D cube points from annotation
        points = transformations.calc_cube_points(annotation_translation=annotation_translation,
                                                  annotation_rotation=annotation_rotation,
                                                  annotation_scale=annotation_scale,
                                                  apply_rotation=True)
        # get all images of relevant frame
        frame_images = lidar_video_content.get('frames', list())[frame_num].get('images', list())
        # iterate over images that correspond with frame and create cube annotation for each image if it is inside
        # the image boundaries
        self.iterate_image(cameras=camera_calibrations,
                           images=frame_images,
                           points=points,
                           annotation=annotation)

    def project_annotations_to_2d(self, item: dl.Item):
        """
        Function that projects annotations to 2D from the original lidar scene annotations.
        :param item: DL lidar scene item
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
        for annotation in annotations:
            end_frame = annotation.metadata.get('system', dict()).get('endFrame')
            last_projected_frame = 0
            # extract first frame annotation and project it to 2D
            annotation_translation, annotation_scale, annotation_rotation = self.get_annotation_metrics(
                geo=annotation.geo)
            self.calculate_frame_annotations(annotation=annotation,
                                             annotation_translation=annotation_translation,
                                             annotation_rotation=annotation_rotation,
                                             annotation_scale=annotation_scale,
                                             lidar_video_content=lidar_video_content,
                                             camera_calibrations=camera_calibrations,
                                             frame_num=0)

            # iterate over all snapshots and project them to 2D
            annotation_snapshots = annotation.metadata.get('system', dict()).get('snapshots_', list())
            for snapshot in annotation_snapshots:
                last_snapshot_frame = snapshot.get('frame')
                # if there are frames between the last projected frame and the current snapshot frame
                # project them to 2D with the last projected frame metrics
                if last_snapshot_frame != last_projected_frame + 1:
                    for frame_num in range(last_projected_frame + 1, last_snapshot_frame):
                        self.calculate_frame_annotations(annotation=annotation,
                                                         annotation_translation=annotation_translation,
                                                         annotation_rotation=annotation_rotation,
                                                         annotation_scale=annotation_scale,
                                                         lidar_video_content=lidar_video_content,
                                                         camera_calibrations=camera_calibrations,
                                                         frame_num=frame_num)
                # project snapshot to 2D with the snapshot metrics
                annotation_translation, annotation_scale, annotation_rotation = self.get_metrics_snapshot(
                    snapshot.get('data', dict()))
                self.calculate_frame_annotations(annotation=annotation,
                                                 annotation_translation=annotation_translation,
                                                 annotation_rotation=annotation_rotation,
                                                 annotation_scale=annotation_scale,
                                                 lidar_video_content=lidar_video_content,
                                                 camera_calibrations=camera_calibrations,
                                                 frame_num=last_snapshot_frame)
                last_projected_frame = last_snapshot_frame
            # if there are frames between the last projected frame and the end frame
            # project them to 2D with the last projected frame metrics
            if last_projected_frame < end_frame:
                for frame_num in range(last_projected_frame + 1, end_frame):
                    self.calculate_frame_annotations(annotation=annotation,
                                                     annotation_translation=annotation_translation,
                                                     annotation_rotation=annotation_rotation,
                                                     annotation_scale=annotation_scale,
                                                     lidar_video_content=lidar_video_content,
                                                     camera_calibrations=camera_calibrations,
                                                     frame_num=frame_num)


if __name__ == "__main__":
    runner = AnnotationProjection()
    # frames json item ID
    item = dl.items.get(item_id='')
    runner.project_annotations_to_2d(item=item)
