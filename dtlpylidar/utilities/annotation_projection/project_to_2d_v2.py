import dtlpy as dl
import os
import numpy as np
import uuid
import json
import dtlpylidar.utilities.transformations as transformations
from tqdm import tqdm
import cv2
import math
from scipy.ndimage import map_coordinates


class AnnotationProjection(dl.BaseServiceRunner):
    def __init__(self):
        ...

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

    @staticmethod
    def get_front_and_back_faces(points_3d, annotation_pixels):
        """
        Determine front and back faces using the projection distance to the camera plane.
        """
        # Face definitions based on your corner layout (8 corners)
        face_indices = {
            "front": [4, 5, 7, 6],  # +Z
            "back": [0, 1, 3, 2],  # -Z
            "left": [0, 2, 6, 4],
            "right": [1, 3, 7, 5],
            "top": [2, 3, 7, 6],
            "bottom": [0, 1, 5, 4]
        }

        camera_forward = np.array([0, 0, 1])  # Z-axis in camera space

        # Score each face by how close it is to the image plane (dot with Z axis)
        face_scores = {
            name: np.mean(points_3d[indices] @ camera_forward)
            for name, indices in face_indices.items()
        }

        # Closest = front, farthest = back
        sorted_faces = sorted(face_scores.items(), key=lambda x: x[1])
        front_face_name = sorted_faces[0][0]
        back_face_name = sorted_faces[-1][0]

        front_indices = face_indices[front_face_name]
        back_indices = face_indices[back_face_name]

        front_points_2d = annotation_pixels[front_indices]
        back_points_2d = annotation_pixels[back_indices]

        # Sort points top-left, top-right, bottom-right, bottom-left
        front = AnnotationProjection.sort_face(front_points_2d)
        back = AnnotationProjection.sort_face(back_points_2d)

        return front, back

    def create_annotation(self, option, label, points_3d, annotation_pixels, width, height, full_annotations_only):
        """
        Create annotation from 3D cube 8 points projected on 2D image.
        :param option: annotation type, can be "Cube", "Polygons", or "Points".
        :param label: annotation label
        :param points_3d: 3D cube points in camera space (PCD normalized).
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

        # Use camera-plane projection distance to get correct faces
        front, back = self.get_front_and_back_faces(points_3d=points_3d, annotation_pixels=annotation_pixels)

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
            cube = dl.Cube(
                label=label,
                front_tl=front_tl,
                front_tr=front_tr,
                front_br=front_br,
                front_bl=front_bl,
                back_tl=back_tl,
                back_tr=back_tr,
                back_br=back_br,
                back_bl=back_bl
            )
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

    def handle_frame(self, labels_colors, cameras_map, frame_images, frame_annotations, flags):
        """
        Calculate frame annotations.
        Iterate over images that correspond with frame and create cube annotation for each image if it is inside the image boundaries.
        :param labels_colors: map of label names to colors
        :param cameras_map: map of camera IDs to camera calibrations
        :param frame_images: images that correspond with the current frame number
        :param frame_annotations: annotations that correspond with the current frame number
        :param flags: flags for the projection
        :return: None
        """
        # Parse flags
        full_annotations_only = flags.get("full_annotations_only", False)
        project_remotely = flags.get("project_remotely", False)
        support_external_parameters = flags.get("support_external_parameters", True)
        apply_image_undistortion = flags.get("apply_image_undistortion", False)
        apply_annotation_distortion = flags.get("apply_annotation_distortion", True)

        # Debug flags:
        # "Manual"
        # "OpenCV" (Debug)
        undistort_mode = "Manual"
        projection_mode = "Manual"

        # iterate over images that correspond with frame
        images_map = {}
        for idx, image_calibrations in enumerate(frame_images):
            print("Processing image {}/{}...".format(idx + 1, len(frame_images)))

            ###############
            # Extract MVP #
            ###############

            item_id = image_calibrations.get('image_id')
            item = dl.items.get(item_id=item_id)
            images_map[item_id] = {
                "item": item
            }

            # Set builder
            images_map[item_id]["builder"] = item.annotations.builder()

            # Get image and camera calibrations
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
            k1 = camera_distortion.get("k1", 0.0)
            k2 = camera_distortion.get("k2", 0.0)
            k3 = camera_distortion.get("k3", 0.0)
            k4 = camera_distortion.get("k4", 0.0)
            k5 = camera_distortion.get("k5", 0.0)
            k6 = camera_distortion.get("k6", 0.0)
            k7 = camera_distortion.get("k7", 0.0)
            k8 = camera_distortion.get("k8", 0.0)
            p1 = camera_distortion.get("p1", 0.0)
            p2 = camera_distortion.get("p2", 0.0)
            xi = camera_distortion.get('xi', 0.0)
            r0 = camera_distortion.get('r0', 0.0)

            # Camera Options:
            camera_model_options = [
                "Regular", # Regular (OpenCV Regular camera)
                "Brown",   # Brown–Conrady
                "Fisheye", # Fisheye (OpenCV Fisheye camera)
                "Kannala", # Kannala-Brandt
                "MEI",     # MEI (KITTI-360 Fisheye cameras: https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/helpers/project.py)
                "Custom",  # Custom
            ]
            camera_model = camera_distortion.get('model', 'Regular')
            # camera_model = 'Custom'

            ################
            # Undistortion #
            ################

            if project_remotely is True:
                # No undistortion support for remote image
                images_map[item_id]["path"] = None

            elif project_remotely is False:
                # Set image paths
                image_path = str(os.path.join(frames_item.id, item.filename[1:]))
                img_name, img_ext = os.path.splitext(image_path)
                output_image_path = f"{img_name}_annotated{img_ext}"
                images_map[item_id]["path"] = image_path
                images_map[item_id]["output_path"] = output_image_path

                if not os.path.exists(image_path):
                    download_image_path = os.path.dirname(image_path)
                    item.download(local_path=download_image_path)

                # Remove distortion from image
                if apply_image_undistortion:
                    # Manual Undistortion
                    if undistort_mode == "Manual":
                        h, w = item.height, item.width
                        map_x = np.zeros((h, w), dtype=np.float32)
                        map_y = np.zeros((h, w), dtype=np.float32)

                        if camera_model == "Regular":
                            for j in range(h):
                                for i in range(w):
                                    # Normalize
                                    y = (j - cy) / fy
                                    x = (i - cx - skew * y) / fx

                                    # Radial distortion coefficients
                                    r = math.sqrt(x * x + y * y)

                                    r2 = r * r
                                    r4 = r2 * r2
                                    r6 = r4 * r2

                                    radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
                                    x_r = x * radial
                                    y_r = y * radial

                                    # Tangent distortion coefficients
                                    x_d = x_r + (2.0 * p1 * x_r * y_r + p2 * (r2 + 2.0 * x_r * x_r))
                                    y_d = y_r + (p1 * (r2 + 2.0 * y_r * y_r) + 2.0 * p2 * x_r * y_r)

                                    map_x[j, i] = fx * x_d + cx
                                    map_y[j, i] = fy * y_d + cy

                        elif camera_model == "Brown":
                            for j in range(h):
                                for i in range(w):
                                    # Normalize
                                    y = (j - cy) / fy
                                    x = (i - cx - skew * y) / fx

                                    # Radial distortion coefficients
                                    r = math.sqrt(x * x + y * y)

                                    r2 = r * r
                                    r4 = r2 * r2
                                    r6 = r4 * r2
                                    r8 = r6 * r2
                                    r10 = r8 * r2
                                    r12 = r10 * r2
                                    r14 = r12 * r2
                                    r16 = r14 * r2

                                    radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8 + k5 * r10 + k6 * r12 + k7 * r14 + k8 * r16
                                    x_r = x * radial
                                    y_r = y * radial

                                    # Tangent distortion coefficients
                                    x_d = x_r + (2.0 * p1 * x_r * y_r + p2 * (r2 + 2.0 * x_r * x_r))
                                    y_d = y_r + (p1 * (r2 + 2.0 * y_r * y_r) + 2.0 * p2 * x_r * y_r)

                                    map_x[j, i] = fx * x_d + cx
                                    map_y[j, i] = fy * y_d + cy

                        elif camera_model == "Fisheye":
                            for j in range(h):
                                for i in range(w):
                                    # Normalize
                                    y = (j - cy) / fy
                                    x = (i - cx - skew * y) / fx

                                    # Radial distortion coefficients
                                    r = math.sqrt(x * x + y * y)
                                    theta = np.arctan(r)

                                    theta2 = theta * theta
                                    theta4 = theta2 * theta2
                                    theta6 = theta4 * theta2
                                    theta8 = theta6 * theta2

                                    radial = theta * (1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8)
                                    scale = radial / r if r > 1e-8 else 1.0
                                    x_d = scale * x
                                    y_d = scale * y

                                    map_x[j, i] = fx * x_d + cx
                                    map_y[j, i] = fy * y_d + cy

                        elif camera_model == "Kannala":
                            for j in range(h):
                                for i in range(w):
                                    # Normalize
                                    y = (j - cy) / fy
                                    x = (i - cx - skew * y) / fx

                                    # Radial distortion coefficients
                                    r = math.sqrt(x * x + y * y)
                                    theta = np.arctan(r)

                                    theta2 = theta * theta
                                    theta4 = theta2 * theta2
                                    theta6 = theta4 * theta2
                                    theta8 = theta6 * theta2
                                    theta10 = theta8 * theta2
                                    theta12 = theta10 * theta2
                                    theta14 = theta12 * theta2
                                    theta16 = theta14 * theta2

                                    radial = theta * (1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8 + k5 * theta10 + k6 * theta12 + k7 * theta14 + k8 * theta16)
                                    scale = radial / r if r > 1e-8 else 1.0
                                    x_r = scale * x
                                    y_r = scale * y

                                    # Tangent distortion coefficients
                                    if support_external_parameters:
                                        r2 = x_r * x_r + y_r * y_r
                                        x_d = x_r + (2.0 * p1 * x_r * y_r + p2 * (r2 + 2.0 * x_r * x_r))
                                        y_d = y_r + (p1 * (r2 + 2.0 * y_r * y_r) + 2.0 * p2 * x_r * y_r)
                                    else:
                                        x_d = x_r
                                        y_d = y_r

                                    map_x[j, i] = fx * x_d + cx
                                    map_y[j, i] = fy * y_d + cy

                        elif camera_model == "MEI":
                            for j in range(h):
                                for i in range(w):
                                    # Normalize
                                    z = 1.0
                                    y = (j - cy) / fy
                                    x = (i - cx - skew * y) / fx

                                    norm = float(np.linalg.norm(np.array([x, y, z])))
                                    x = x / norm
                                    y = y / norm
                                    z = z / norm

                                    # Radial distortion coefficients
                                    x /= z + xi
                                    y /= z + xi

                                    r2 = x * x + y * y
                                    radial = 1.0 + k1 * r2 + k2 * r2 * r2
                                    x_r = x * radial
                                    y_r = y * radial
                                    # z_r = norm * point_3d[:, 2] / np.abs(point_3d[:, 2])

                                    # Tangent distortion coefficients
                                    if support_external_parameters:
                                        x_d = x_r + (2.0 * p1 * x_r * y_r + p2 * (r2 + 2.0 * x_r * x_r))
                                        y_d = y_r + (p1 * (r2 + 2.0 * y_r * y_r) + 2.0 * p2 * x_r * y_r)
                                    else:
                                        x_d = x_r
                                        y_d = y_r

                                    # Project back to image plane
                                    map_x[j, i] = fx * x_d + cx
                                    map_y[j, i] = fy * y_d + cy

                        elif camera_model == "Custom":
                            for j in range(h):
                                for i in range(w):
                                    # Normalize
                                    z = 1.0
                                    y = (j - cy) / fy
                                    x = (i - cx - skew * y) / fx

                                    n2 = x * x + y * y
                                    r2 = n2 + z * z
                                    invR = 1.0 / np.sqrt(r2) if r2 != 0.0 else 0.0
                                    invN = 1.0 / np.sqrt(n2) if n2 != 0.0 else 0.0

                                    # Angular projection
                                    theta = np.arccos(z * invR)
                                    xu = theta * x * invN
                                    yu = theta * y * invN
                                    ru2 = xu * xu + yu * yu
                                    ru = np.sqrt(ru2)
                                    ru0 = ru - r0
                                    ru02 = ru0 * ru0

                                    # Radial distortion
                                    fD = 1.0
                                    for i, ki in enumerate([k1, k2, k3, k4, k5, k6, k7, k8]):
                                        if ki != 0.0:
                                            fD += ki * ru02 ** (i + 1)

                                    xd = xu * fD
                                    yd = yu * fD

                                    # Tangential distortion
                                    ydT = (yd + 2.0 * p1 * xu * yu + p2 * (ru2 + 2.0 * yu * yu))
                                    xdT = (xd + 2.0 * p2 * xu * yu + p1 * (ru2 + 2.0 * xu * xu))

                                    # Map to pixel coordinates
                                    x_d = xdT
                                    y_d = ydT

                                    # Project back to image plane
                                    map_x[j, i] = fx * x_d + cx
                                    map_y[j, i] = fy * y_d + cy

                        else:
                            raise ValueError(
                                f"Unsupported camera model: {camera_model}. "
                                f"Supported models are: {camera_model_options}."
                            )

                        image = cv2.imread(image_path)
                        coords = [map_y.ravel(), map_x.ravel()]
                        undistorted_r = map_coordinates(
                            image[:, :, 0], coords, order=1, mode='reflect').reshape((h, w))
                        undistorted_g = map_coordinates(
                            image[:, :, 1], coords, order=1, mode='reflect').reshape((h, w))
                        undistorted_b = map_coordinates(
                            image[:, :, 2], coords, order=1, mode='reflect').reshape((h, w))
                        undistorted = np.stack(
                            [undistorted_r, undistorted_g, undistorted_b], axis=2).astype(np.uint8)
                        cv2.imwrite(output_image_path, undistorted)

                    # OpenCV Undistortion
                    elif undistort_mode == "OpenCV":
                        # Distortion coefficients
                        D = np.array([k1, k2, p1, p2, k3], dtype=np.float64)

                        # Original distorted image
                        image = cv2.imread(image_path)
                        h, w = image.shape[:2]

                        # Compute optimal rectified camera matrix (keeps FOV)
                        new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))

                        # Undistort
                        undistorted = cv2.undistort(image, K, D, None, new_K)

                        # Save or display
                        x, y, w, h = roi
                        undistorted = undistorted[y:y + h, x:x + w]
                        cv2.imwrite(output_image_path, undistorted)

                    else:
                        raise ValueError(
                            f"Unsupported undistort mode: {undistort_mode}. "
                            f"Supported modes are 'Manual' and 'OpenCV'."
                        )
                else:
                    # Overwrite annotated image
                    image = cv2.imread(image_path)
                    cv2.imwrite(output_image_path, image)

                images_map[item_id] = {
                    "item": item,
                    "path": image_path,
                    "output_path": output_image_path
                }

            else:
                raise ValueError("project_remotely must be either True or False.")

            ##########################
            # Apply MVP + Distortion #
            ##########################

            for annotation_data in tqdm(frame_annotations):
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
                    annotation_pixels = []
                    for point_3d in points_3d:
                        (x, y, z) = point_3d
                        if apply_annotation_distortion:
                            if camera_model == "Regular":
                                # Normalize
                                z = z if z != 0 else 1e-8
                                x = x / z
                                y = y / z

                                # Radial distortion coefficients
                                r = math.sqrt(x * x + y * y)

                                r2 = r * r
                                r4 = r2 * r2
                                r6 = r4 * r2

                                radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
                                x_r = x * radial
                                y_r = y * radial

                                # Tangent distortion coefficients
                                x_d = x_r + (2.0 * p1 * x_r * y_r + p2 * (r2 + 2.0 * x_r * x_r))
                                y_d = y_r + (p1 * (r2 + 2.0 * y_r * y_r) + 2.0 * p2 * x_r * y_r)

                            elif camera_model == "Brown":
                                # Normalize
                                z = z if z != 0 else 1e-8
                                x = x / z
                                y = y / z

                                # Radial distortion coefficients
                                r = math.sqrt(x * x + y * y)

                                r2 = r * r
                                r4 = r2 * r2
                                r6 = r4 * r2
                                r8 = r6 * r2
                                r10 = r8 * r2
                                r12 = r10 * r2
                                r14 = r12 * r2
                                r16 = r14 * r2

                                radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8 + k5 * r10 + k6 * r12 + k7 * r14 + k8 * r16
                                x_r = x * radial
                                y_r = y * radial

                                # Tangent distortion coefficients
                                x_d = x_r + (2.0 * p1 * x_r * y_r + p2 * (r2 + 2.0 * x_r * x_r))
                                y_d = y_r + (p1 * (r2 + 2.0 * y_r * y_r) + 2.0 * p2 * x_r * y_r)

                            elif camera_model == "Fisheye":
                                # Radial distortion coefficients
                                r = math.sqrt(x * x + y * y)
                                theta = np.arccos(z / math.sqrt(x * x + y * y + z * z))

                                theta2 = theta * theta
                                theta4 = theta2 * theta2
                                theta6 = theta4 * theta2
                                theta8 = theta6 * theta2

                                radial = theta * (1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8)
                                scale = radial / r if r > 1e-8 else 1.0
                                x_d = scale * x
                                y_d = scale * y

                            elif camera_model == "Kannala":
                                # Radial distortion coefficients
                                r = math.sqrt(x * x + y * y)
                                theta = np.arccos(z / math.sqrt(x * x + y * y + z * z))

                                theta2 = theta * theta
                                theta4 = theta2 * theta2
                                theta6 = theta4 * theta2
                                theta8 = theta6 * theta2
                                theta10 = theta8 * theta2
                                theta12 = theta10 * theta2
                                theta14 = theta12 * theta2
                                theta16 = theta14 * theta2

                                radial = theta * (1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8 + k5 * theta10 + k6 * theta12 + k7 * theta14 + k8 * theta16)
                                scale = radial / r if r > 1e-8 else 1.0
                                x_r = scale * x
                                y_r = scale * y

                                # Tangent distortion coefficients
                                if support_external_parameters:
                                    r2 = x_r * x_r + y_r * y_r
                                    x_d = x_r + (2.0 * p1 * x_r * y_r + p2 * (r2 + 2.0 * x_r * x_r))
                                    y_d = y_r + (p1 * (r2 + 2.0 * y_r * y_r) + 2.0 * p2 * x_r * y_r)
                                else:
                                    x_d = x_r
                                    y_d = y_r

                            elif camera_model == "MEI":
                                # Normalize
                                norm = float(np.linalg.norm(np.array([x, y, z])))
                                x = x / norm
                                y = y / norm
                                z = z / norm

                                # Radial distortion coefficients
                                x /= z + xi
                                y /= z + xi

                                r2 = x * x + y * y
                                radial = 1.0 + k1 * r2 + k2 * r2 * r2
                                x_r = x * radial
                                y_r = y * radial
                                # z_r = norm * point_3d[:, 2] / np.abs(point_3d[:, 2])

                                # Tangent distortion coefficients
                                if support_external_parameters:
                                    x_d = x_r + (2.0 * p1 * x_r * y_r + p2 * (r2 + 2.0 * x_r * x_r))
                                    y_d = y_r + (p1 * (r2 + 2.0 * y_r * y_r) + 2.0 * p2 * x_r * y_r)
                                else:
                                    x_d = x_r
                                    y_d = y_r

                            elif camera_model == "Custom":
                                # Normalize
                                n2 = x * x + y * y
                                r2 = n2 + z * z
                                invR = 1.0 / np.sqrt(r2) if r2 != 0.0 else 0.0
                                invN = 1.0 / np.sqrt(n2) if n2 != 0.0 else 0.0

                                # Angular projection
                                theta = np.arccos(z * invR)
                                xu = theta * x * invN
                                yu = theta * y * invN
                                ru2 = xu * xu + yu * yu
                                ru = np.sqrt(ru2)
                                ru0 = ru - r0
                                ru02 = ru0 * ru0

                                # Radial distortion
                                fD = 1.0
                                for i, ki in enumerate([k1, k2, k3, k4, k5, k6, k7, k8]):
                                    if ki != 0.0:
                                        fD += ki * ru02 ** (i + 1)

                                xd = xu * fD
                                yd = yu * fD

                                # Tangential distortion
                                ydT = (yd + 2.0 * p1 * xu * yu + p2 * (ru2 + 2.0 * yu * yu))
                                xdT = (xd + 2.0 * p2 * xu * yu + p1 * (ru2 + 2.0 * xu * xu))

                                # Map to pixel coordinates
                                x_d = xdT
                                y_d = ydT

                            else:
                                raise ValueError(
                                    f"Unsupported camera model: {camera_model}.\n"
                                    f"Supported models are: {camera_model_options}"
                                )
                        else:
                            # If no distortion, just use the projected pixel directly
                            x_d = x / z
                            y_d = y / z

                        # Convert back to pixel coordinates
                        mv_points = np.array([x_d, y_d, 1, 1])
                        mvp_points = projection_matrix @ mv_points
                        annotation_pixels.append(mvp_points[:2])

                    annotation_pixels = np.array(annotation_pixels)

                # OpenCV MVP
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

                    if apply_annotation_distortion:
                        # 2D camera #
                        if camera_model == "Regular":
                            D = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
                            (points_2d, _) = cv2.projectPoints(object_points, rvec, tvec, K, D)
                        elif camera_model == "Fisheye":
                            D = np.array([k1, k2, k3, k4], dtype=np.float64)
                            (points_2d, _) = cv2.fisheye.projectPoints(object_points, rvec, tvec, K, D)
                        else:
                            raise ValueError(
                                f"Unsupported camera model: {camera_model}.\n"
                                f"Supported models are 'Regular' and 'Fisheye'."
                            )
                    else:
                        D = np.zeros((5,), dtype=np.float64)
                        (points_2d, _) = cv2.projectPoints(object_points, rvec, tvec, K, D)

                    # points_2d: (N, 1, 2) - OpenCV format
                    annotation_pixels = points_2d.reshape(-1, 2)  # (N, 2)

                # Select annotation option based on the projection mode
                if project_remotely:
                    if apply_annotation_distortion:
                        option = "Polygons"
                    else:
                        option = "Cube"
                else:
                    option = "Points"

                # create annotation if it is inside the image boundaries
                annotation_definitions = self.create_annotation(
                    option=option,
                    label=annotation_data["label"],
                    points_3d=points_3d,
                    annotation_pixels=annotation_pixels,
                    width=item.width,
                    height=item.height,
                    full_annotations_only=full_annotations_only
                )
                # if cube annotation is not None add it to the item
                if annotation_definitions is None:
                    continue

                if project_remotely is True:
                    # Add annotation to the item builder
                    for annotation_definition in annotation_definitions:
                        images_map[item_id]["builder"].add(
                            annotation_definition=annotation_definition,
                            object_id=annotation_data["object_id"],
                            object_visible=annotation_data["object_visible"]
                        )
                else:
                    annotation_points_2d = []
                    for annotation_definition in annotation_definitions:
                        if isinstance(annotation_definition, dl.Cube):
                            annotation_definition_geo = [
                                annotation_definition.front_tl,
                                annotation_definition.front_tr,
                                annotation_definition.front_br,
                                annotation_definition.front_bl,
                                annotation_definition.back_tl,
                                annotation_definition.back_tr,
                                annotation_definition.back_br,
                                annotation_definition.back_bl
                            ]
                        else:
                            annotation_definition_geo = annotation_definition.geo
                        annotation_points_2d.append(annotation_definition_geo)
                    image_path = images_map.get(item_id, dict()).get("output_path")
                    image = cv2.imread(image_path)

                    edges = [
                        (0, 1), (1, 2), (2, 3), (3, 0),  # front face
                        (4, 5), (5, 6), (6, 7), (7, 4),  # back face
                        (0, 4), (1, 5), (2, 6), (3, 7)  # connecting edges
                    ]

                    for start_idx, end_idx in edges:
                        pt1 = tuple(np.round(annotation_points_2d[start_idx]).astype(int))
                        pt2 = tuple(np.round(annotation_points_2d[end_idx]).astype(int))
                        color = labels_colors.get(annotation_data["label"], (255, 255, 255))  # Default color is white if label not found
                        cv2.line(image, pt1, pt2, color=color, thickness=2)
                    cv2.imwrite(image_path, image)

        # Upload annotation to the item
        if project_remotely is True:
            for item_id in images_map.keys():
                images_map[item_id]["builder"].upload()

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

    def project_annotations_to_2d(self, item: dl.Item, flags: dict):
        """
        Function that projects annotations to 2D from the original lidar scene annotations.
        :param item: DL lidar scene item
        :param flags: dictionary with flags:
        - full_annotations_only: if True, only full annotations will be projected to 2D
        - project_remotely: if True, annotations will be uploaded to the image items, otherwise annotations will be drawn on the images locally.
        - support_external_parameters: if True, support external parameters for the projection (Like: k4, k5, k6, k7, k8)
        - apply_image_undistortion: if True, apply image undistortion to the images before projection
        - apply_annotation_distortion: if True, apply annotation distortion to the projected pixels
        :return: None
        """
        # Get labels colors
        def hex_to_bgr(hex_color: str):
            hex_color = hex_color.lstrip('#')
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            return b, g, r  # OpenCV uses BGR

        labels_colors = {}
        dataset = item.dataset
        for label_name, label_data in dataset.labels_flat_dict.items():
            labels_colors[label_name] = hex_to_bgr(label_data.color)

        # Download lidar scene video's json
        # items_path = os.path.join(os.getcwd(), "data", uid)
        items_path = os.path.join(os.getcwd(), "data", item.id)
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
        cameras_map = {camera.get('id'): camera for camera in camera_calibrations}
        frames_count = len(lidar_video_content.get('frames', list()))
        for frame_num in range(frames_count):
            print("Frame number:", frame_num)

            #################
            # Handle Images #
            #################
            frame_images = lidar_video_content.get('frames', list())[frame_num].get('images', list())
            frame_annotations = frame_annotations_per_frame.get(frame_num, list())
            self.handle_frame(
                labels_colors=labels_colors,
                cameras_map=cameras_map,
                frame_images=frame_images,
                frame_annotations=frame_annotations,
                flags=flags
            )


if __name__ == "__main__":
    # frames json item ID
    dl.setenv('rc')
    item_id = '685822032da29c6039f80472'
    frames_item = dl.items.get(item_id=item_id)
    flags = dict(
        full_annotations_only=False,
        project_remotely=False,
        support_external_parameters=True,
        apply_image_undistortion=False,
        apply_annotation_distortion=True
    )

    runner = AnnotationProjection()
    runner.project_annotations_to_2d(
        item=frames_item,
        flags=flags
    )
