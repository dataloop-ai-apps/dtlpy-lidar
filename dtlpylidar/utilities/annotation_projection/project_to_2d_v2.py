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
            return b, g, r  # OpenCV uses BGR

        self.labels_colors = {}
        if dataset is not None:
            for label_name, label_data in dataset.labels_flat_dict.items():
                self.labels_colors[label_name] = hex_to_bgr(label_data.color)

    def create_annotation(self, option, label, annotation_pixels, depths, width, height, full_annotations_only):
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

        front_indices = np.argsort(depths)[:4]
        back_indices = np.argsort(depths)[4:]

        front_points = annotation_pixels[front_indices]
        back_points = annotation_pixels[back_indices]

        front = self.sort_face(front_points)
        back = self.sort_face(back_points)

        # Assign
        front_tl = front["tl"]
        front_tr = front["tr"]
        front_br = front["br"]
        front_bl = front["bl"]

        back_tl = back["tl"]
        back_tr = back["tr"]
        back_br = back["br"]
        back_bl = back["bl"]

        if option == "Cube":
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
            return [cube]

        elif option == "Polygons":
            # create polygons annotation
            polygon1 = dl.Polygon(
                geo=[
                    [front_tl[0], front_tl[1]],  # front top left
                    [front_tr[0], front_tr[1]],  # front top right
                    [front_br[0], front_br[1]],  # front bottom right
                    [front_bl[0], front_bl[1]],  # front bottom left
                ],
                label=label
            )
            polygon2 = dl.Polygon(
                geo=[
                    [back_tl[0], back_tl[1]],    # back top left
                    [back_tr[0], back_tr[1]],    # back top right
                    [back_br[0], back_br[1]],    # back bottom right
                    [back_bl[0], back_bl[1]]     # back bottom left
                ],
                label=label
            )
            return [polygon1, polygon2]
        elif option == "Points":
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
        else:
            raise ValueError(f"Unsupported option: {option}. Supported options are 'Cube', 'Polygons', and 'Points'.")

    @staticmethod
    def sort_face(points_4):
        # points_4: (4, 2) array
        points_4 = np.array(points_4)
        sorted_by_y = points_4[np.argsort(points_4[:, 1])]  # sort top to bottom

        top_two = sorted_by_y[:2]
        bottom_two = sorted_by_y[2:]

        # now sort left/right within top and bottom
        tl, tr = top_two[np.argsort(top_two[:, 0])]
        bl, br = bottom_two[np.argsort(bottom_two[:, 0])]

        return {
            "tl": tuple(tl),
            "tr": tuple(tr),
            "br": tuple(br),
            "bl": tuple(bl)
        }

    # TODO: remove factor_m at the end
    def calculate_frame_annotations(self, annotation_data,
                                    frame_images, images_map, cameras_map,
                                    factor_m, # TODO: Remove later
                                    full_annotations_only, apply_annotation_distortion,
                                    project_remotely, support_external_parameters):
        """
        Calculate frame annotations.
        Iterate over images that correspond with frame and create cube annotation for each image if it is inside the image boundaries.
        :param annotation_data: annotation data from frame_annotations_per_frame
        :param frame_images: images that correspond with the current frame number
        :param images_map: map of image IDs to image paths
        :param cameras_map: map of camera IDs to camera calibrations
        :param full_annotations_only: if True, only full annotations will be projected to 2D
        :param apply_annotation_distortion: if True, apply annotation distortion to the projected pixels
        :param project_remotely: if True, annotations will be uploaded to the image items, otherwise annotations will be drawn on the images locally.
        :param support_external_parameters: if True, support external parameters for the projection (k4, k5, k6, k7, k8)
        :return: None
        """
        projection_mode = "Manual" # "Manual" or "OpenCV"

        # Cube annotation data geo
        annotation_translation = annotation_data["geo"][0]
        annotation_scale = annotation_data["geo"][1]
        annotation_rotation = annotation_data["geo"][2]

        # calculate 3D cube points from annotation (PCD normalized)
        points = transformations.calc_cuboid_corners(
            dimensions=annotation_scale
        )
        model_matrix = transformations.calc_transform_matrix(
            rotation=annotation_rotation,
            position=annotation_translation
        )

        # iterate over images that correspond with frame
        for idx, image_calibrations in enumerate(frame_images):
            # get image and camera calibrations
            item_id = image_calibrations.get('image_id')
            item = images_map[item_id]["item"]
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
            cx = intrinsic_data.get('cx', 0.0)
            cy = intrinsic_data.get('cy', 0.0)
            skew = intrinsic_data.get('skew', 0.0)
            projection_matrix = np.array([
                [fx, skew, cx, 0],
                [0 , fy  , cy, 0],
                [0 , 0   , 1 , 0],
                [0 , 0   , 0 , 1]
            ])

            camera_distortion = intrinsic_data.get('distortion', dict())
            # factor_m = camera_distortion.get('m', 1.0)
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

            # Manual MVP
            if projection_mode == "Manual":
                mv = view_matrix @ model_matrix
                points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])  # (N, 4)
                points_4d = (mv @ points_homogeneous.T).T  # (N, 4)
                points_3d = points_4d[:, :3] / np.abs(points_4d[:, 3:4])  # (N, 3)

                # Check if the points are behind the camera
                if not np.all(points_3d[:, 2] > 0):
                    continue  # Skip if any point is behind the camera

                # Distortion

                points_2d = points_3d[:, :2] / points_3d[:, 2:3]  # (N, 2)
                annotation_pixels = []
                for point_2d in points_2d:
                    (x, y) = point_2d
                    if apply_annotation_distortion:
                        # TODO: find a flag to support switch
                        camera_mode = "Kannala" # "2D" or "Fisheye" or "MEI" or "Kannala"

                        # 2D
                        if camera_mode == "2D":
                            if support_external_parameters:
                                r = math.sqrt(x * x + y * y) / r0
                            else:
                                r = math.sqrt(x * x + y * y)

                            r2 = r * r # if k1 != 0.0 else 0
                            r4 = r2 * r2 # if k2 != 0.0 else 0
                            r6 = r4 * r2 # if k3 != 0.0 else 0
                            r8 = r6 * r2 # if k4 != 0.0 else 0
                            r10 = r8 * r2 # if k5 != 0.0 else 0
                            r12 = r10 * r2 # if k6 != 0.0 else 0
                            r14 = r12 * r2 # if k7 != 0.0 else 0
                            r16 = r14 * r2 # if k8 != 0.0 else 0

                            if support_external_parameters:
                                radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8 + k5 * r10 + k6 * r12 + k7 * r14 + k8 * r16
                            else:
                                radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
                            x_d = x * radial + (2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x))
                            y_d = y * radial + (p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y)

                        # Fisheye camera #
                        elif camera_mode == "Fisheye":
                            if support_external_parameters:
                                r = math.sqrt(x * x + y * y) / r0
                            else:
                                r = math.sqrt(x * x + y * y)
                            theta = math.atan(r)

                            theta2 = theta * theta  # if k1 != 0.0 else 0
                            theta4 = theta2 * theta2  # if k2 != 0.0 else 0
                            theta6 = theta4 * theta2  # if k3 != 0.0 else 0
                            theta8 = theta6 * theta2  # if k4 != 0.0 else 0
                            theta10 = theta8 * theta2  # if k5 != 0.0 else 0
                            theta12 = theta10 * theta2  # if k6 != 0.0 else 0
                            theta14 = theta12 * theta2  # if k7 != 0.0 else 0
                            theta16 = theta14 * theta2  # if k8 != 0.0 else 0

                            if support_external_parameters:
                                radial = theta * (1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8 + k5 * theta10 + k6 * theta12 + k7 * theta14 + k8 * theta16)
                            else:
                                radial = theta * (1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8)

                            scale = radial / r if r > 1e-8 else 1.0
                            x_d = scale * x
                            y_d = scale * y

                        elif camera_mode == "MEI":
                            xi = camera_distortion.get('xi', 1.0)
                            d1 = x ** 2 + y ** 2 + 1
                            d2 = xi * math.sqrt(d1) + 1
                            x_d = x / d2
                            y_d = y / d2

                            if support_external_parameters:
                                r2 = (x_d ** 2 + y_d ** 2) / (r0 ** 2)
                            else:
                                r2 = x_d ** 2 + y_d ** 2
                            x_d += (2 * p1 * x_d * y_d + p2 * (r2 + 2 * x_d ** 2))
                            y_d += (p1 * (r2 + 2 * y_d ** 2) + 2 * p2 * x_d * y_d)

                        elif camera_mode == "Kannala":
                            if support_external_parameters:
                                r = math.sqrt(x * x + y * y) # / r0
                            else:
                                r = math.sqrt(x * x + y * y)
                            theta = math.atan(r)

                            # Compute distortion terms using theta powers
                            theta2 = theta * theta
                            theta4 = theta2 * theta2
                            theta6 = theta4 * theta2
                            theta8 = theta6 * theta2
                            theta10 = theta8 * theta2
                            theta12 = theta10 * theta2
                            theta14 = theta12 * theta2
                            theta16 = theta14 * theta2

                            if support_external_parameters:
                                radial = theta * (1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8 + k5 * theta10 + k6 * theta12 + k7 * theta14 + k8 * theta16)
                            else:
                                radial = theta + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8

                            scale = radial / r if r > 1e-8 else 1.0
                            x_d = scale * x
                            y_d = scale * y

                            if support_external_parameters:
                                r2 = x_d ** 2 + y_d ** 2
                                x_d += 2.0 * p1 * x_d * y_d + p2 * (r2 + 2.0 * x_d ** 2)
                                y_d += p1 * (r2 + 2.0 * y_d ** 2) + 2.0 * p2 * x_d * y_d

                        else:
                            raise ValueError(
                                f"Unsupported camera mode: {camera_mode}. "
                                f"Supported modes are '2D', 'Fisheye', 'MEI', and 'Kannala'."
                            )
                    else:
                        # If no distortion, just use the projected pixel directly
                        x_d = x
                        y_d = y

                    # Convert back to pixel coordinates
                    mv_points = np.array([x_d, y_d, 1, 1])
                    mvp_points = projection_matrix @ mv_points
                    annotation_pixels.append(mvp_points[:2])

                annotation_pixels = np.array(annotation_pixels)

            # MVP OPENCV
            else:
                if support_external_parameters:
                    raise ValueError("OpenCV projection mode does not support external parameters.")

                mv = view_matrix @ model_matrix  # Model View matrix
                K = projection_matrix[:3, :3]  # Projection matrix

                # Check if the points are behind the camera
                points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])  # (N, 4)
                points_4d = (mv @ points_homogeneous.T).T  # (N, 4)
                points_3d = points_4d[:, :3] / np.abs(points_4d[:, 3:4])  # (N, 3)
                if not np.all(points_3d[:, 2] > 0):
                    continue  # Skip if any point is behind the camera

                # Option 1 - Apply MV on points manually
                # points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])  # (N, 4)
                # points_4d = (mv @ points_homogeneous.T).T  # (N, 4)
                # points_3d = points_4d[:, :3]  # (N, 3)
                # object_points = points_3d.reshape(-1, 3)  # (N, 3)
                # rvec = np.zeros((3, 1), dtype=np.float64)
                # tvec = np.zeros((3, 1), dtype=np.float64)

                # Option 2 - Apply MV on points using OpenCV
                object_points = points.reshape(1, -1, 3)
                rvec = cv2.Rodrigues(mv[:3, :3])[0].astype(np.float64)  # Rotation vector
                tvec = mv[:3, 3].reshape(-1, 1).astype(np.float64)  # Translation vector

                # 2D camera #
                # if apply_annotation_distortion:
                #     D = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
                # else:
                #     D = np.zeros((5,), dtype=np.float64)
                # points_2d, _ = cv2.projectPoints(object_points, rvec, tvec, K, D)
                # annotation_pixels = points_2d.reshape(-1, 2)  # (N, 2)

                # Fisheye camera #
                if apply_annotation_distortion:
                    D = np.array([k1, k2, k3, k4], dtype=np.float64)
                else:
                    D = np.zeros((4,), dtype=np.float64)
                points_2d, _ = cv2.fisheye.projectPoints(object_points, rvec, tvec, K, D)
                annotation_pixels = points_2d.reshape(-1, 2)  # (N, 2)

            if project_remotely:
                if apply_annotation_distortion:
                    option = "Polygons"
                else:
                    option = "Cube"
            else:
                option = "Points"

            # Find Front and Back points (Front = points with smaller Z - closer to camera)
            depths = points_3d[:, 2]  # Camera-space Z

            # create annotation if it is inside the image boundaries
            annotations = self.create_annotation(
                option=option,
                label=annotation_data["label"],
                annotation_pixels=annotation_pixels,
                depths=depths,
                width=item.width,
                height=item.height,
                full_annotations_only=full_annotations_only
            )
            # if cube annotation is not None add it to the item
            if annotations is None:
                continue

            if project_remotely is True:
                # TODO: TBD - aggregated same image annotations and upload in one go

                # create annotation builder
                builder = item.annotations.builder()
                # add annotation to the item
                for annotation in annotations:
                    builder.add(
                        annotation_definition=annotation,
                        object_id=annotation_data["object_id"],
                        object_visible=annotation_data["object_visible"]
                    )
                # upload annotation to the item
                item.annotations.upload(builder)
            else:
                anno_points_2d = []
                for annotation in annotations:
                    anno_points_2d.append(annotation.geo)
                image_path = images_map.get(item_id, dict()).get("output_path")
                image = cv2.imread(image_path)

                edges = [
                    (0, 1), (1, 2), (2, 3), (3, 0),  # front face
                    (4, 5), (5, 6), (6, 7), (7, 4),  # back face
                    (0, 4), (1, 5), (2, 6), (3, 7)  # connecting edges
                ]

                for start_idx, end_idx in edges:
                    pt1 = tuple(np.round(anno_points_2d[start_idx]).astype(int))
                    pt2 = tuple(np.round(anno_points_2d[end_idx]).astype(int))
                    color = self.labels_colors.get(annotation_data["label"], (255, 255, 255))  # Default color is white if label not found
                    cv2.line(image, pt1, pt2, color=color, thickness=2)
                cv2.imwrite(image_path, image)


    @staticmethod
    def build_frame_annotations_per_frame_mapping(lidar_video_content, annotations):
        # Order annotations frames by frame number
        frames_count = len(lidar_video_content.get('frames', list()))
        annotation: dl.Annotation
        frame_annotations_per_frame = {frame_number: list() for frame_number in range(frames_count)}
        for annotation in annotations:
            start_frame = annotation.metadata.get('system', dict()).get('frame')
            end_frame = annotation.metadata.get('system', dict()).get('endFrame')
            snapshots = annotation.metadata.get('system', dict()).get('snapshots_', list())

            # Annotation data
            annotation_data = dict(
                object_id=annotation.object_id,
                geo=annotation.geo,
                label=annotation.label,
                object_visible=annotation.object_visible
            )
            frame_annotations_per_frame[start_frame].append(annotation_data)
            last_keyframe_frame = start_frame

            # If there are snapshots, add them to the frame annotations
            for snapshot in snapshots:
                frame_annotation: dl.entities.FrameAnnotation = dl.entities.FrameAnnotation.from_snapshot(
                    annotation=annotation,
                    _json=snapshot,
                    fps=None
                )
                last_snapshot_frame = frame_annotation.frame_num

                # if there are frames between the last projected frame and the current snapshot frame
                # project them to 2D with the last projected frame metrics
                if last_snapshot_frame != last_keyframe_frame + 1:
                    for frame_num in range(last_keyframe_frame + 1, last_snapshot_frame):
                        frame_annotations_per_frame[frame_num].append(annotation_data)

                # project snapshot to 2D with the snapshot metrics
                annotation_data = dict(
                    object_id=annotation.object_id,
                    geo=frame_annotation.geo,
                    label=frame_annotation.label,
                    object_visible=frame_annotation.object_visible
                )
                frame_annotations_per_frame[last_snapshot_frame].append(annotation_data)
                last_keyframe_frame = last_snapshot_frame

            if last_keyframe_frame < end_frame:
                for frame_num in range(last_keyframe_frame + 1, end_frame):
                    frame_annotations_per_frame[frame_num].append(annotation_data)

        return frame_annotations_per_frame

    def project_annotations_to_2d(self, item: dl.Item,
                                  full_annotations_only: bool = False,
                                  project_remotely: bool = True,
                                  support_external_parameters: bool = False):
        """
        Function that projects annotations to 2D from the original lidar scene annotations.
        :param item: DL lidar scene item
        :param full_annotations_only: if True, only full annotations will be projected to 2D
        :param project_remotely: if True, annotations will be uploaded to the image items, otherwise annotations will be drawn on the images locally.
        :return: None
        """
        # TODO: Debug
        apply_image_undistortion = False
        apply_annotation_distortion = True
        # factor_m = -150.0
        factor_m = 1

        # Download lidar scene video's json
        items_path = os.path.join(os.getcwd(), item.id)
        frames_item_path = item.download(local_path=items_path, overwrite=True)
        with open(frames_item_path, 'r') as f:
            lidar_video_content = json.load(f)

        # Get all annotations of the item
        filters_annotation = dl.Filters(resource=dl.FiltersResource.ANNOTATION)
        filters_annotation.add(field='type', values='cube_3d')
        annotations = item.annotations.list(filters=filters_annotation)
        if isinstance(annotations, dl.entities.PagedEntities):
            annotations = list(annotations.all())

        frame_annotations_per_frame = self.build_frame_annotations_per_frame_mapping(
            lidar_video_content=lidar_video_content,
            annotations=annotations
        )

        # get all camera calibrations
        camera_calibrations = lidar_video_content.get('cameras', list())
        frames_count = len(lidar_video_content.get('frames', list()))
        for frame_num in range(frames_count):
            print("Frame number:", frame_num)

            # TODO: Debug
            if frame_num != 0:
                exit()

            #################
            # Handle Images #
            #################

            frame_images = lidar_video_content.get('frames', list())[frame_num].get('images', list())
            cameras_map = {camera.get('id'): camera for camera in camera_calibrations}

            images_map = {}
            for idx, image_calibrations in enumerate(frame_images):
                # get image and camera calibrations
                item_id = image_calibrations.get('image_id')
                item = dl.items.get(item_id=item_id)
                images_map[item_id] = {
                    "item": item
                }
                if project_remotely is True:
                    images_map[item_id]["path"] = None
                elif project_remotely is False:
                    image_path = str(os.path.join(frames_item.id, item.filename[1:]))
                    images_map[item_id]["path"] = image_path
                    if not os.path.exists(image_path):
                        download_image_path = os.path.dirname(image_path)
                        item.download(local_path=download_image_path)

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
                                [fx, s, cx],
                                [0, fy, cy],
                                [0, 0, 1]
                            ])

                            camera_distortion = intrinsic_data.get('distortion', dict())
                            k1 = camera_distortion["k1"]
                            k2 = camera_distortion["k2"]
                            k3 = camera_distortion["k3"]
                            p1 = camera_distortion["p1"]
                            p2 = camera_distortion["p2"]
                            # factor_m = camera_distortion.get('m', 1.0)
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

                    # Overwrite annotated image
                    image_path = str(os.path.join(frames_item.id, item.filename[1:]))
                    image = cv2.imread(image_path)
                    img_name, img_ext = os.path.splitext(image_path)
                    output_image_path = f"{img_name}_annotated{img_ext}"
                    cv2.imwrite(output_image_path, image)

                    images_map[item_id] = {
                        "item": item,
                        "path": image_path,
                        "output_path": output_image_path
                    }
                else:
                    raise ValueError("project_remotely must be either True or False.")

            frame_annotations = frame_annotations_per_frame.get(frame_num, list())
            for annotation_data in tqdm(frame_annotations):
                self.calculate_frame_annotations(
                    annotation_data=annotation_data,
                    frame_images=frame_images,
                    images_map=images_map,
                    cameras_map=cameras_map,
                    factor_m=factor_m,  # TODO: Remove later
                    full_annotations_only=full_annotations_only,
                    apply_annotation_distortion=apply_annotation_distortion,
                    project_remotely=project_remotely,
                    support_external_parameters=support_external_parameters
                )


if __name__ == "__main__":
    # frames json item ID
    dl.setenv('rc')
    item_id = '68415b9d1bd0d57f611190a1'
    frames_item = dl.items.get(item_id=item_id)
    full_annotations_only = False
    project_remotely = False
    support_external_parameters = True

    runner = AnnotationProjection(dataset=frames_item.dataset)
    runner.project_annotations_to_2d(
        item=frames_item,
        full_annotations_only=full_annotations_only,
        project_remotely=project_remotely,
        support_external_parameters=support_external_parameters
    )
