import dtlpy as dl
import os
import numpy as np
import uuid
import json
import dtlpylidar.utilities.transformations as transformations
from tqdm import tqdm
import cv2
import math


class AnnotationProjection(dl.BaseServiceRunner):
    def __init__(self, dataset: dl.Dataset = None):
        def hex_to_bgr(hex_color: str):
            hex_color = hex_color.lstrip('#')
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            return (b, g, r)  # OpenCV uses BGR

        self.labels_colors = {}
        if dataset is not None:
            for label_name, label_data in dataset.labels_flat_dict.items():
                self.labels_colors[label_name] = hex_to_bgr(label_data.color)

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

        # points_dict = {
        #     "front_tl": front_tl,
        #     "front_tr": front_tr,
        #     "front_bl": front_bl,
        #     "front_br": front_br,
        #     "back_tl": back_tl,
        #     "back_tr": back_tr,
        #     "back_bl": back_bl,
        #     "back_br": back_br
        # }

        # # create cube annotation
        # cube = dl.Cube(label=label,
        #                front_tl=front_tl,
        #                front_tr=front_tr,
        #                front_br=front_br,
        #                front_bl=front_bl,
        #                back_tl=back_tl,
        #                back_tr=back_tr,
        #                back_br=back_br,
        #                back_bl=back_bl)
        # return [cube]

        # create polygons annotation
        # polygon1 = dl.Polygon(
        #     geo=[
        #         [front_tl[0], front_tl[1]],  # front top left
        #         [front_tr[0], front_tr[1]],  # front top right
        #         [front_br[0], front_br[1]],  # front bottom right
        #         [front_bl[0], front_bl[1]],  # front bottom left
        #     ],
        #     label=label
        # )
        # polygon2 = dl.Polygon(
        #     geo=[
        #         [back_tl[0], back_tl[1]],    # back top left
        #         [back_tr[0], back_tr[1]],    # back top right
        #         [back_br[0], back_br[1]],    # back bottom right
        #         [back_bl[0], back_bl[1]]     # back bottom left
        #     ],
        #     label=label
        # )
        # return [polygon1, polygon2]

        # create points annotation
        points = [
            dl.Point(x=front_tl[0], y=front_tl[1], label=label),  # front top left
            dl.Point(x=front_tr[0], y=front_tr[1], label=label),  # front top right
            dl.Point(x=front_br[0], y=front_br[1], label=label),  # front bottom right
            dl.Point(x=front_bl[0], y=front_bl[1], label=label),  # front bottom left
            dl.Point(x=back_tl[0], y=back_tl[1], label=label),    # back top left
            dl.Point(x=back_tr[0], y=back_tr[1], label=label),    # back top right
            dl.Point(x=back_br[0], y=back_br[1], label=label),    # back bottom right
            dl.Point(x=back_bl[0], y=back_bl[1], label=label)     # back bottom left
        ]
        return points

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
        apply_image_undistortion = False
        apply_annotation_distortion = True
        factor_m = -150.0

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

        images_map = {}
        for idx, image_calibrations in enumerate(frame_images):
            # get image and camera calibrations
            item_id = image_calibrations.get('image_id')
            item = dl.items.get(item_id=item_id)
            image_path = str(os.path.join(frames_item.id, item.name))
            if not os.path.exists(image_path):
                item.download(local_path=frames_item.id)

                # Remove distortion from image
                if apply_image_undistortion:
                    camera_id = image_calibrations.get('camera_id')
                    camera_calibrations = cameras_map.get(camera_id)
                    sensors_data = camera_calibrations.get('sensorsData')

                    # calculate projection matrix (Default values: Orthographic projection)
                    intrinsic_data = sensors_data.get('intrinsicData', dict())
                    fx = intrinsic_data.get('fx', 1.0)
                    fy = intrinsic_data.get('fy', 1.0)
                    s = intrinsic_data.get('skew', 0.0)
                    cx = intrinsic_data.get('cx', 0.0)
                    cy = intrinsic_data.get('cy', 0.0)
                    K = np.array([
                        [fx, s , cx],
                        [0 , fy, cy],
                        [0 , 0 , 1 ]
                    ])

                    camera_distortion = intrinsic_data.get('distortion', dict())
                    k1 = camera_distortion["k1"]
                    k2 = camera_distortion["k2"]
                    k3 = camera_distortion["k3"]
                    p1 = camera_distortion["p1"]
                    p2 = camera_distortion["p2"]
                    D = factor_m * np.array([k1, k2, p1, p2, k3], dtype=np.float64)  # Distortion coefficients

                    # Original distorted image
                    image = cv2.imread(image_path)
                    h, w = image.shape[:2]

                    # Compute optimal rectified camera matrix (keeps FOV)
                    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=0)

                    # Undistort
                    undistorted = cv2.undistort(image, K, D, None, new_K)

                    # Save or display
                    cv2.imwrite(image_path, undistorted)

            images_map[item_id] = image_path

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

            camera_distortion = intrinsic_data.get('distortion', dict())
            k1 = camera_distortion["k1"] * factor_m
            k2 = camera_distortion["k2"] * factor_m
            k3 = camera_distortion["k3"] * factor_m
            k4 = camera_distortion.get("k4", 0.0) * factor_m  # Optional, if not present, set to 0
            k5 = camera_distortion.get("k5", 0.0) * factor_m  # Optional, if not present, set to 0
            k6 = camera_distortion.get("k6", 0.0) * factor_m  # Optional, if not present, set to 0
            k7 = camera_distortion.get("k7", 0.0) * factor_m  # Optional, if not present, set to 0
            k8 = camera_distortion.get("k8", 0.0) * factor_m  # Optional, if not present, set to 0
            p1 = camera_distortion["p1"] * factor_m
            p2 = camera_distortion["p2"] * factor_m
            r0 = camera_distortion.get('r0', 1.0)

            # MANUAL projection

            mvp = projection_matrix @ view_matrix @ model_matrix
            points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])  # (N, 4)
            projected_points = (mvp @ points_homogeneous.T).T  # (N, 4)
            projected_pixels = projected_points[:, :2] / np.abs(projected_points[:, 2:3])  # (N, 2)

            points_2d = []
            for projected_pixel in projected_pixels:
                x_px, y_px = projected_pixel

                if apply_annotation_distortion:
                    # Normalized #

                    # Normalize to camera coordinates
                    x = (x_px - cx) / fx
                    y = (y_px - cy) / fy

                    r = math.sqrt(x ** 2 + y ** 2) / r0
                    r2 = (r ** 2) if k1 != 0.0 else 0
                    r4 = (r ** 4) if k2 != 0.0 else 0
                    r6 = (r ** 6) if k3 != 0.0 else 0
                    r8 = (r ** 8) if k4 != 0.0 else 0
                    r10 = (r ** 10) if k5 != 0.0 else 0
                    r12 = (r ** 12) if k6 != 0.0 else 0
                    r14 = (r ** 14) if k7 != 0.0 else 0
                    r16 = (r ** 16) if k8 != 0.0 else 0

                    radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8 + k5 * r10 + k6 * r12 + k7 * r14 + k8 * r16

                    x_d = x * radial + (2.0 * p1 * x * y + p2 * (r2 + 2.0 * x ** 2))
                    y_d = y * radial + (p1 * (r2 + 2.0 * y ** 2) + 2.0 * p2 * x * y)

                    # Convert back to pixel coordinates
                    # u = fx * x_d + s * y_d + cx
                    u = fx * x_d + cx
                    v = fy * y_d + cy

                    # # Regular #
                    #
                    # x = x_px
                    # y = y_px
                    #
                    # r = math.sqrt(x ** 2 + y ** 2)
                    # r2 = (r ** 2) if k1 > 0 else 0
                    # r4 = (r ** 4) if k2 > 0 else 0
                    # r6 = (r ** 6) if k3 > 0 else 0
                    # r8 = (r ** 8) if k4 > 0 else 0
                    # r10 = (r ** 10) if k5 > 0 else 0
                    # r12 = (r ** 12) if k6 > 0 else 0
                    # r14 = (r ** 14) if k7 > 0 else 0
                    # r16 = (r ** 16) if k8 > 0 else 0
                    #
                    # radial = 1 + k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8 + k5 * r10 + k6 * r12 + k7 * r14 + k8 * r16
                    #
                    # x_distorted = x * radial + (2 * p1 * x * y + p2 * (r2 + 2 * x ** 2))
                    # y_distorted = y * radial + (p1 * (r2 + 2 * y ** 2) + 2 * p2 * x * y)
                    #
                    # # Convert back to pixel coordinates
                    # u = x_distorted
                    # v = y_distorted
                else:
                    # If no distortion, just use the projected pixel directly
                    u = x_px
                    v = y_px

                points_2d.append([u, v])

            # OPENCV projection

            # mv = view_matrix @ model_matrix  # Model View matrix
            # p = projection_matrix[:3, :3]  # Projection matrix
            # points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])  # (N, 4)
            # projected_points = (mv @ points_homogeneous.T).T  # (N, 4)
            # projected_points = projected_points[:, :3]  # (N, 3)
            #
            # rvec = np.zeros((3, 1), dtype=np.float64)
            # tvec = np.zeros((3, 1), dtype=np.float64)
            # dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
            #
            # # append zeros dim to 3rd coordinate # (N, 2) -> (N, 3)
            # points_2d, _ = cv2.projectPoints(projected_points, rvec, tvec, p, dist_coeffs)
            # points_2d = points_2d.reshape(-1, 2)  # (N, 2)


            # create cube annotation if it is inside the image boundaries
            annotations = self.create_annotation(
                label=label,
                annotation_pixels=points_2d,
                width=item.width,
                height=item.height,
                full_annotations_only=full_annotations_only
            )
            # if cube annotation is not None add it to the item
            if annotations is None:
                continue

            anno_2d = []
            for annotation in annotations:
                anno_2d.append(annotation.geo)
            image_path = images_map.get(item_id)
            image = cv2.imread(image_path)

            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),  # front face
                (4, 5), (5, 6), (6, 7), (7, 4),  # back face
                (0, 4), (1, 5), (2, 6), (3, 7)  # connecting edges
            ]


            for start_idx, end_idx in edges:
                pt1 = tuple(np.round(points_2d[start_idx]).astype(int))
                pt2 = tuple(np.round(points_2d[end_idx]).astype(int))
                color = self.labels_colors.get(label, (255, 255, 255))  # Default color is white if label not found
                cv2.line(image, pt1, pt2, color=color, thickness=2)
            cv2.imwrite(image_path, image)

            # # create annotation builder
            # builder = item.annotations.builder()
            # # add annotation to the item
            # for annotation in annotations:
            #     builder.add(
            #         annotation_definition=annotation,
            #         object_id=object_id,
            #         object_visible=object_visible
            #     )
            # # upload annotation to the item
            # item.annotations.upload(builder)

    def project_annotations_to_2d(self, item: dl.Item, full_annotations_only: bool = False):
        """
        Function that projects annotations to 2D from the original lidar scene annotations.
        :param item: DL lidar scene item
        :param full_annotations_only: if True, only full annotations will be projected to 2D
        :return: None
        """
        # download lidar scene video's json
        # uid = str(uuid.uuid4())
        # items_path = os.path.join(os.getcwd(), uid)
        items_path = os.path.join(os.getcwd(), item.id)
        path = item.download(local_path=items_path, overwrite=True)
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
            start_frame = annotation.metadata.get('system', dict()).get('frame')
            end_frame = annotation.metadata.get('system', dict()).get('endFrame')
            object_id = annotation.object_id

            last_projected_frame = start_frame
            # extract first frame annotation and project it to 2D
            annotation_translation, annotation_scale, annotation_rotation = self.get_annotation_metrics(
                geo=annotation.geo)
            annotation_label = annotation.label
            annotation_object_visible = annotation.object_visible
            self.calculate_frame_annotations(object_id=object_id,
                                             label=annotation_label,
                                             object_visible=annotation_object_visible,
                                             annotation_translation=annotation_translation,
                                             annotation_rotation=annotation_rotation,
                                             annotation_scale=annotation_scale,
                                             lidar_video_content=lidar_video_content,
                                             camera_calibrations=camera_calibrations,
                                             frame_num=last_projected_frame,
                                             full_annotations_only=full_annotations_only)
            # # iterate over all snapshots and project them to 2D
            # annotation_snapshots = annotation.metadata.get('system', dict()).get('snapshots_', list())
            # for snapshot in annotation_snapshots:
            #     frame_annotation: dl.entities.FrameAnnotation = dl.entities.FrameAnnotation.from_snapshot(
            #         annotation=annotation,
            #         _json=snapshot,
            #         fps=None
            #     )
            #     last_snapshot_frame = frame_annotation.frame_num
            #     # if there are frames between the last projected frame and the current snapshot frame
            #     # project them to 2D with the last projected frame metrics
            #     if last_snapshot_frame != last_projected_frame + 1:
            #         for frame_num in range(last_projected_frame + 1, last_snapshot_frame):
            #             self.calculate_frame_annotations(object_id=object_id,
            #                                              label=annotation_label,
            #                                              object_visible=annotation_object_visible,
            #                                              annotation_translation=annotation_translation,
            #                                              annotation_rotation=annotation_rotation,
            #                                              annotation_scale=annotation_scale,
            #                                              lidar_video_content=lidar_video_content,
            #                                              camera_calibrations=camera_calibrations,
            #                                              frame_num=frame_num,
            #                                              full_annotations_only=full_annotations_only)
            #     # project snapshot to 2D with the snapshot metrics
            #     annotation_translation, annotation_scale, annotation_rotation = self.get_annotation_metrics(
            #         geo=frame_annotation.geo)
            #     annotation_label = frame_annotation.label
            #     annotation_object_visible = frame_annotation.object_visible
            #     self.calculate_frame_annotations(object_id=object_id,
            #                                      label=annotation_label,
            #                                      object_visible=annotation_object_visible,
            #                                      annotation_translation=annotation_translation,
            #                                      annotation_rotation=annotation_rotation,
            #                                      annotation_scale=annotation_scale,
            #                                      lidar_video_content=lidar_video_content,
            #                                      camera_calibrations=camera_calibrations,
            #                                      frame_num=last_snapshot_frame,
            #                                      full_annotations_only=full_annotations_only)
            #     last_projected_frame = last_snapshot_frame
            # # if there are frames between the last projected frame and the end frame
            # # project them to 2D with the last projected frame metrics
            # if last_projected_frame < end_frame:
            #     for frame_num in range(last_projected_frame + 1, end_frame):
            #         self.calculate_frame_annotations(object_id=object_id,
            #                                          label=annotation_label,
            #                                          object_visible=annotation_object_visible,
            #                                          annotation_translation=annotation_translation,
            #                                          annotation_rotation=annotation_rotation,
            #                                          annotation_scale=annotation_scale,
            #                                          lidar_video_content=lidar_video_content,
            #                                          camera_calibrations=camera_calibrations,
            #                                          frame_num=frame_num,
            #                                          full_annotations_only=full_annotations_only)


if __name__ == "__main__":
    # frames json item ID
    dl.setenv('rc')
    item_id = '683f24e48b8ed565b78dcdbd'
    frames_item = dl.items.get(item_id=item_id)
    full_annotations_only = False

    runner = AnnotationProjection(dataset=frames_item.dataset)
    runner.project_annotations_to_2d(item=frames_item, full_annotations_only=full_annotations_only)
