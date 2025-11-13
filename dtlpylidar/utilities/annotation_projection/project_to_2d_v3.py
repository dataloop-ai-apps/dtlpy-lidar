import dtlpy as dl
import os
import numpy as np
import json
import dtlpylidar.utilities.transformations as transformations
from tqdm import tqdm
import cv2
import math
from scipy.ndimage import map_coordinates
from enum import Enum
# import uuid


# ============================================================================
# CONSTANTS
# ============================================================================

# Camera Options:
# TODO: Put in ReadMe.md
class CameraModel(str, Enum):
    """Camera model type constants."""
    BC = "bc"            # Brown–Conrady [OpenCV 2D camera] (See: https://boofcv.org/index.php?title=Tutorial_Camera_Calibration#:~:text=.-,Brown%20Model,-The%20Brown%20camera)
    KB = "kb"            # Kannala-Brandt - Symmetric [OpenCV Fisheye camera] (See: https://boofcv.org/index.php?title=Tutorial_Camera_Calibration#:~:text=equations%20to%20Brown.-,Kannala%2DBrandt%20Model,-Kannala%2DBrandt%20%5B3)
    MEI = "mei"          # MEI [Universal Omni Model] (See: https://boofcv.org/index.php?title=Tutorial_Camera_Calibration#:~:text=the%20tangential%20coefficients.-,Universal%20Omni%20Model,-Universal%20Omni%20%5B2)
                         # (Example: KITTI-360 Fisheye cameras: https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/helpers/project.py)
    CUSTOM0 = "custom0"  # Custom0


class AnnotationOption(str, Enum):
    """Annotation option type constants."""
    CUBE = "cube"
    POLYGONS = "polygons"
    POINTS = "points"


# ============================================================================
# CAMERA MODEL HANDLER - Eliminates Code Duplication
# ============================================================================

class CameraModelHandler:
    """
    Centralized camera model distortion and undistortion logic.
    This eliminates ~600 lines of duplicated code.
    """
    
    def __init__(self):
        """Initialize camera model function maps."""
        self.DISTORTION_FUNCTIONS = {
            CameraModel.BC: CameraModelHandler.apply_brown_conrady_distortion,
            CameraModel.KB: CameraModelHandler.apply_kannala_brandt_distortion,
            CameraModel.MEI: CameraModelHandler.apply_mei_distortion,
            CameraModel.CUSTOM0: CameraModelHandler.apply_custom0_distortion,
        }
    
    @staticmethod
    def apply_brown_conrady_distortion(x, y, z, **kwargs):
        """Apply Brown-Conrady camera model distortion."""
        k1 = kwargs.get('k1', 0.0)
        k2 = kwargs.get('k2', 0.0)
        k3 = kwargs.get('k3', 0.0)
        k4 = kwargs.get('k4', 0.0)
        k5 = kwargs.get('k5', 0.0)
        k6 = kwargs.get('k6', 0.0)
        k7 = kwargs.get('k7', 0.0)
        k8 = kwargs.get('k8', 0.0)
        p1 = kwargs.get('p1', 0.0)
        p2 = kwargs.get('p2', 0.0)
        
        z = z if z != 0 else 1e-8
        x = x / z
        y = y / z
        
        # r = math.sqrt(x * x + y * y)
        r2 = x * x + y * y
        
        radial_sum = 1.0
        for idx, ki in enumerate([k1, k2, k3, k4, k5, k6, k7, k8]):
            if ki != 0.0:
                radial_sum += ki * r2 ** (idx + 1)
        
        x_r = x * radial_sum
        y_r = y * radial_sum
        
        if p1 != 0.0 and p2 != 0.0:
            x_t = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
            y_t = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
            x_d = x_r + x_t
            y_d = y_r + y_t
            return x_d, y_d
        else:
            return x_r, y_r
    
    @staticmethod
    def apply_kannala_brandt_distortion(x, y, z, **kwargs):
        """Apply Kannala-Brandt (Symmetric) camera model distortion."""
        k1 = kwargs.get('k1', 0.0)
        k2 = kwargs.get('k2', 0.0)
        k3 = kwargs.get('k3', 0.0)
        k4 = kwargs.get('k4', 0.0)
        k5 = kwargs.get('k5', 0.0)
        k6 = kwargs.get('k6', 0.0)
        k7 = kwargs.get('k7', 0.0)
        k8 = kwargs.get('k8', 0.0)
        
        z = z if z != 0 else 1e-8
        x = x / z
        y = y / z

        r = math.sqrt(x * x + y * y)
        theta = math.atan(r)
        theta2 = theta * theta
        
        radial_sum = 1.0
        for idx, ki in enumerate([k1, k2, k3, k4, k5, k6, k7, k8]):
            if ki != 0.0:
                radial_sum += ki * theta2 ** (idx + 1)
        
        radial = theta * radial_sum
        scale = radial / r if r > 1e-8 else 1.0
        x_r = x * scale
        y_r = y * scale
        
        return x_r, y_r
    
    @staticmethod
    def apply_mei_distortion(x, y, z, **kwargs):
        """Apply MEI camera model distortion."""
        k1 = kwargs.get('k1', 0.0)
        k2 = kwargs.get('k2', 0.0)
        p1 = kwargs.get('p1', 0.0)
        p2 = kwargs.get('p2', 0.0)
        xi = kwargs.get('xi', 0.0)
        
        norm = float(np.linalg.norm(np.array([x, y, z])))
        x = x / norm
        y = y / norm
        z = z / norm
        
        x /= z + xi
        y /= z + xi
        
        # r = math.sqrt(x * x + y * y)
        r2 = x * x + y * y
        radial_sum = 1.0
        for idx, ki in enumerate([k1, k2]):
            if ki != 0.0:
                radial_sum += ki * r2 ** (idx + 1)
        
        x_r = x * radial_sum
        y_r = y * radial_sum
        
        if p1 != 0.0 and p2 != 0.0:
            x_t = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
            y_t = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
            x_d = x_r + x_t
            y_d = y_r + y_t
            return x_d, y_d
        else:
            return x_r, y_r
    
    @staticmethod
    def apply_custom0_distortion(x, y, z, **kwargs):
        """Apply Custom0 camera model distortion."""
        k1 = kwargs.get('k1', 0.0)
        k2 = kwargs.get('k2', 0.0)
        k3 = kwargs.get('k3', 0.0)
        k4 = kwargs.get('k4', 0.0)
        k5 = kwargs.get('k5', 0.0)
        k6 = kwargs.get('k6', 0.0)
        k7 = kwargs.get('k7', 0.0)
        k8 = kwargs.get('k8', 0.0)
        p1 = kwargs.get('p1', 0.0)
        p2 = kwargs.get('p2', 0.0)
        r0 = kwargs.get('r0', 0.0)
        
        n2 = x * x + y * y
        r2 = n2 + z * z
        invR = 1.0 / np.sqrt(r2) if r2 != 0.0 else 0.0
        invN = 1.0 / np.sqrt(n2) if n2 != 0.0 else 0.0
        
        theta = np.arccos(z * invR)
        xu = theta * x * invN
        yu = theta * y * invN
        ru2 = xu * xu + yu * yu
        ru = np.sqrt(ru2)
        ru0 = ru - r0
        ru02 = ru0 * ru0
        
        fD = 1.0
        for idx, ki in enumerate([k1, k2, k3, k4, k5, k6, k7, k8]):
            if ki != 0.0:
                fD += ki * ru02 ** (idx + 1)
        
        x_r = xu * fD
        y_r = yu * fD
        
        x_d = x_r + (2.0 * p1 * xu * yu + p2 * (ru2 + 2.0 * xu * xu))
        y_d = y_r + (p1 * (ru2 + 2.0 * yu * yu) + 2.0 * p2 * xu * yu)
        return x_d, y_d
    
    def apply_distortion_to_point(self, x, y, z, camera_distortion):
        """Apply distortion to a single point based on camera model."""
        x_d, y_d = self.DISTORTION_FUNCTIONS[camera_distortion["model"]](
            x=x, y=y, z=z, **camera_distortion
        )
        return x_d, y_d


# ============================================================================
# MVP CALCULATOR
# ============================================================================

class MVPCalculator:
    """Handles Model-View-Projection matrix calculations."""
    
    @staticmethod
    def calculate_view_matrix(camera_rotation, camera_translation):
        """Calculate view matrix from camera pose."""
        view_matrix = transformations.calc_transform_matrix(
            rotation=camera_rotation,
            position=camera_translation
        )
        return np.linalg.inv(view_matrix)
    
    @staticmethod
    def calculate_projection_matrix(fx, fy, cx, cy, skew):
        """Calculate projection matrix from intrinsic parameters."""
        return np.array([
            [fx, skew, cx, 0],
            [0,  fy,   cy, 0],
            [0,  0,    1,  0],
            [0,  0,    0,  1]
        ])


# ============================================================================
# MAIN CLASS
# ============================================================================

class AnnotationProjection(dl.BaseServiceRunner):
    def __init__(self):
        self.camera_model_handler = CameraModelHandler()
        self.mvp_calculator = MVPCalculator()
        self.face_indices = {
            "front": [4, 5, 7, 6],  # Z = +1
            "back": [0, 1, 3, 2],   # Z = -1
            "left": [0, 2, 6, 4],   # X = -1
            "right": [1, 3, 7, 5],  # X = +1
            "top": [2, 3, 7, 6],    # Y = +1
            "bottom": [0, 1, 5, 4]  # Y = -1
        }
        self.opposite_faces = {
            "front": "back",
            "back": "front",
            "left": "right",
            "right": "left",
            "top": "bottom",
            "bottom": "top"
        }

    @staticmethod
    def intersect_ray_with_face(ray_origin, ray_dir, face_points):
        """
        Intersect a ray with a quad face.
        Returns (distance, intersection_point) if inside face, else None.
        """
        p0, p1, p2, p3 = face_points
        v1 = p1 - p0
        v2 = p3 - p0
        normal = np.cross(v1, v2)
        normal = normal / (np.linalg.norm(normal) + 1e-8)

        denom = np.dot(normal, ray_dir)
        if abs(denom) < 1e-6:
            return None  # ray is parallel to face

        d = np.dot(normal, p0 - ray_origin) / denom
        if d < 0:
            return None  # intersection is behind the camera

        intersection = ray_origin + d * ray_dir

        # Point-in-quad test (inside both triangles)
        def inside_triangle(pt, a, b, c):
            v0 = c - a
            v1 = b - a
            v2 = pt - a
            dot00 = np.dot(v0, v0)
            dot01 = np.dot(v0, v1)
            dot02 = np.dot(v0, v2)
            dot11 = np.dot(v1, v1)
            dot12 = np.dot(v1, v2)
            inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01 + 1e-8)
            u = (dot11 * dot02 - dot01 * dot12) * inv_denom
            v = (dot00 * dot12 - dot01 * dot02) * inv_denom
            return (u >= 0) and (v >= 0) and (u + v <= 1)

        inside = (
            inside_triangle(intersection, p0, p1, p2)
            or inside_triangle(intersection, p0, p2, p3)
        )
        if inside:
            return d, intersection
        else:
            return None

    def get_front_face_by_intersection(self, points_3d):
        camera_origin = np.array([0.0, 0.0, 0.0])
        cube_center = np.mean(points_3d, axis=0)
        ray_dir = cube_center - camera_origin
        ray_dir /= np.linalg.norm(ray_dir)

        closest_face = None
        min_distance = float('inf')

        for name, indices in self.face_indices.items():
            face_pts = points_3d[indices]
            result = self.intersect_ray_with_face(camera_origin, ray_dir, face_pts)
            if result:
                dist, _ = result
                if dist < min_distance:
                    min_distance = dist
                    closest_face = name

        return closest_face  # e.g. "front"

    def get_front_and_back_faces_by_ray(self, points_3d, annotation_pixels):
        # Find the front face by ray intersection
        front_name = self.get_front_face_by_intersection(points_3d)
        back_name = self.opposite_faces[front_name]

        front_indices = self.face_indices[front_name]
        back_indices = self.face_indices[back_name]

        front_points = annotation_pixels[front_indices]
        back_points = annotation_pixels[back_indices]

        front = {
            "tl": tuple(front_points[0]),
            "tr": tuple(front_points[1]),
            "br": tuple(front_points[2]),
            "bl": tuple(front_points[3])
        }
        back = {
            "tl": tuple(back_points[0]),
            "tr": tuple(back_points[1]),
            "br": tuple(back_points[2]),
            "bl": tuple(back_points[3])
        }
        return front, back

    def create_annotation(self, option, label, points_3d, annotation_pixels, width, height, full_annotations_only):
        """
        Create annotation from 3D cube 8 points projected on 2D image.
        :param option: annotation type, from AnnotationOption enum.
        :param label: annotation label
        :param points_3d: 3D cube points in camera space (PCD normalized).
        :param annotation_pixels: annotation 3D cube 8 points projected on 2D image.
        :param width: image width
        :param height: image height
        :param full_annotations_only: if True, only full annotations will be projected to 2D
        :return: cube 3D annotation representation if at least 2 points are inside the image boundaries.
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
        front, back = self.get_front_and_back_faces_by_ray(points_3d=points_3d, annotation_pixels=annotation_pixels)

        # Assign
        front_tl = front["tl"]
        front_tr = front["tr"]
        front_br = front["br"]
        front_bl = front["bl"]

        back_tl = back["tl"]
        back_tr = back["tr"]
        back_br = back["br"]
        back_bl = back["bl"]

        # TODO: Open ticket Feature Request - Bend Cuboid
        if option == AnnotationOption.CUBE:
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
            cubes = [cube]
            return cubes

        elif option == AnnotationOption.POLYGONS:
            # Option 1 - Front & Back Polygons
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
            # polygons = [polygon1, polygon2]

            # Option 2 - Convex Hull Polygon
            pts = annotation_pixels.astype(dtype=np.float32)
            hull = cv2.convexHull(pts)
            polygon = dl.Polygon(
                geo=hull.squeeze().tolist(),
                label=label
            )
            polygons = [polygon]
            return polygons

        elif option == AnnotationOption.POINTS:
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
            raise ValueError(f"Unsupported option: {option}. Supported options are {list(AnnotationOption)}.")

    def handle_frame(self, items_path, labels_colors, cameras_map, frame_images, frame_annotations, config):
        """
        Calculate frame annotations.
        Iterate over images that correspond with frame and create cube annotation for each image if it is inside the image boundaries.
        :param items_path: path to the items directory
        :param labels_colors: map of label names to colors
        :param cameras_map: map of camera IDs to camera calibrations
        :param frame_images: images that correspond with the current frame number
        :param frame_annotations: annotations that correspond with the current frame number
        :param config: config for the projection
        :return: None
        """
        # Parse config
        full_annotations_only = config.get("full_annotations_only", False)
        debug = config.get("debug", False)
        apply_image_undistortion = config.get("apply_image_undistortion", False)
        apply_annotation_distortion = config.get("apply_annotation_distortion", True)

        # Debug flags:
        # "Manual"
        # "OpenCV" (Debug)
        # TODO: Moving OpenCV usage to tests
        undistort_mode = "Manual"
        projection_mode = "Manual"

        # iterate over images that correspond with frame
        images_map = {}
        for idx, image_calibrations in enumerate(frame_images):
            # if idx != 0:
            #     continue
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
            view_matrix = self.mvp_calculator.calculate_view_matrix(camera_rotation, camera_translation)

            # calculate projection matrix (Default values: Orthographic projection)
            intrinsic_data = sensors_data.get('intrinsicData', dict())
            fx = intrinsic_data.get('fx', 1.0)
            fy = intrinsic_data.get('fy', 1.0)
            cx = intrinsic_data.get('cx', 0.0)
            cy = intrinsic_data.get('cy', 0.0)
            skew = intrinsic_data.get('skew', 0.0)
            projection_matrix = self.mvp_calculator.calculate_projection_matrix(fx, fy, cx, cy, skew)

            camera_distortion = intrinsic_data.get('distortion', dict())

            # Default Camera Model
            if "model" not in camera_distortion:
                camera_distortion["model"] = CameraModel.BC
                # camera_distortion["model"] = CameraModel.CUSTOM0
            else:
                camera_model = camera_distortion["model"]
                if camera_model not in list(CameraModel):
                    raise ValueError(
                        f"Unsupported camera model: {camera_model}. "
                        f"Supported models are: {list(CameraModel)}."
                    )
                
                if camera_model == CameraModel.MEI and (undistort_mode == "OpenCV" or projection_mode == "OpenCV"):
                    try:
                        import cv2.omnidir
                    except ImportError:
                        raise ImportError(
                            "cv2.omnidir is not installed. Please install it using 'pip install opencv-contrib-python'."
                        )

            ################
            # Undistortion #
            ################

            # TODO: Replace name to Debug + make both functionality work togther.
            if debug is False:
                # No undistortion support for remote image
                images_map[item_id]["path"] = None

            else:
                # Set image paths
                image_path = str(os.path.join(items_path, item.filename[1:]))
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
                        for j in range(h):
                            for i in range(w):
                                z = 1.0
                                y = (j - cy) / fy
                                x = (i - cx - skew * y) / fx

                                x_d, y_d = self.camera_model_handler.apply_distortion_to_point(
                                    x=x, y=y, z=z, camera_distortion=camera_distortion
                                )
                                
                                map_x[j, i] = fx * x_d + skew * y_d + cx
                                map_y[j, i] = fy * y_d + cy
                        
                        image = cv2.imread(image_path)

                        # Option 1: Using map_coordinates (slower)
                        # coords = [map_y.ravel(), map_x.ravel()]
                        # undistorted_r = map_coordinates(
                        #     image[:, :, 0], coords, order=1, mode='reflect').reshape((h, w))
                        # undistorted_g = map_coordinates(
                        #     image[:, :, 1], coords, order=1, mode='reflect').reshape((h, w))
                        # undistorted_b = map_coordinates(
                        #     image[:, :, 2], coords, order=1, mode='reflect').reshape((h, w))
                        # undistorted = np.stack(
                        #     [undistorted_r, undistorted_g, undistorted_b], axis=2).astype(np.uint8)

                        # Option 2: Using remap (faster)
                        undistorted = cv2.remap(
                            image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
                        )

                    # OpenCV Undistortion
                    elif undistort_mode == "OpenCV":
                        camera_model = camera_distortion["model"]
                        k1 = camera_distortion.get("k1", 0.0)
                        k2 = camera_distortion.get("k2", 0.0)
                        k3 = camera_distortion.get("k3", 0.0)
                        k4 = camera_distortion.get("k4", 0.0)
                        p1 = camera_distortion.get("p1", 0.0)
                        p2 = camera_distortion.get("p2", 0.0)
                        xi = camera_distortion.get("xi", 0.0)

                        # Original distorted image
                        image = cv2.imread(image_path)
                        h, w = image.shape[:2]

                        # Build K matrix for OpenCV
                        K = np.array([
                            [fx, skew, cx],
                            [0, fy, cy],
                            [0, 0, 1]
                        ])

                        # Distortion coefficients
                        if camera_model == CameraModel.BC:
                            D = np.array([k1, k2, p1, p2, k3], dtype=np.float64)

                            # Compute optimal rectified camera matrix (keeps FOV)
                            new_K, roi = cv2.getOptimalNewCameraMatrix(
                                cameraMatrix=K, distCoeffs=D, imageSize=(w, h), alpha=1, newImgSize=(w, h)
                            )

                            # Undistort
                            undistorted = cv2.undistort(
                                src=image, cameraMatrix=K, distCoeffs=D, dst=None, newCameraMatrix=new_K
                            )
                            x, y, w, h = roi
                            undistorted = undistorted[y:y + h, x:x + w]
                        elif camera_model == CameraModel.KB:
                            D = np.array([k1, k2, k3, k4], dtype=np.float64)

                            Knew = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                                K=K, D=D, image_size=(w, h), R=np.eye(3), P=None, balance=0.0, new_size=(w, h), fov_scale=1.0
                            )
                            undistorted = cv2.fisheye.undistortImage(image, K, D, None, Knew, (w, h))
                        elif camera_model == CameraModel.MEI:
                            D = np.array([k1, k2, k3, k4], dtype=np.float64)

                            Knew = K
                            undistorted = cv2.omnidir.undistortImage(
                                distorted=image, K=K, D=D, xi=xi, flags=cv2.omnidir.RECTIFY_PERSPECTIVE, undistorted=None, Knew=Knew, new_size=(w, h), R=None
                            )
                        else:
                            raise ValueError(
                                f"[OpenCV] Unsupported camera model: {camera_model}. "
                                f"Supported models are: {CameraModel.BC} and {CameraModel.KB}."
                            )

                    else:
                        raise ValueError(
                            f"Unsupported undistort mode: {undistort_mode}. "
                            f"Supported modes are 'Manual' and 'OpenCV'."
                        )
                    
                    # Save locally
                    cv2.imwrite(output_image_path, undistorted)

                else:
                    # Overwrite annotated image
                    image = cv2.imread(image_path)
                    cv2.imwrite(output_image_path, image)

                images_map[item_id] = {
                    "item": item,
                    "path": image_path,
                    "output_path": output_image_path
                }

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

                # TODO: Validate new radial and tangent fixes - Compare to OpenCV
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
                    if apply_annotation_distortion:
                        projection_function = self.camera_model_handler.apply_distortion_to_point
                    else:
                        # If no distortion, just use the projected pixel directly
                        projection_function = lambda x, y, z, camera_distortion: (x / z, y / z)
                    
                    annotation_pixels = []
                    for point_3d in points_3d:
                        (x, y, z) = point_3d
                        x_d, y_d = projection_function(
                            x=x, y=y, z=z, camera_distortion=camera_distortion
                        )
                        # Convert back to pixel coordinates
                        mv_points = np.array([x_d, y_d, 1, 1])
                        mvp_points = projection_matrix @ mv_points
                        annotation_pixels.append(mvp_points[:2])

                    annotation_pixels = np.array(annotation_pixels)

                # OpenCV MVP
                elif projection_mode == "OpenCV":
                    camera_model = camera_distortion["model"]
                    k1 = camera_distortion.get("k1", 0.0)
                    k2 = camera_distortion.get("k2", 0.0)
                    k3 = camera_distortion.get("k3", 0.0)
                    k4 = camera_distortion.get("k4", 0.0)
                    p1 = camera_distortion.get("p1", 0.0)
                    p2 = camera_distortion.get("p2", 0.0)
                    xi = camera_distortion.get("xi", 0.0)

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
                        if camera_model == CameraModel.BC:
                            D = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
                            (points_2d, _) = cv2.projectPoints(object_points, rvec, tvec, K, D)
                        elif camera_model == CameraModel.KB:
                            D = np.array([k1, k2, k3, k4], dtype=np.float64)
                            (points_2d, _) = cv2.fisheye.projectPoints(object_points, rvec, tvec, K, D)
                        elif camera_model == CameraModel.MEI:
                            D = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
                            (points_2d, _) = cv2.omnidir.projectPoints(object_points, rvec, tvec, K, xi, D)
                        else:
                            raise ValueError(
                                f"[OpenCV] Unsupported camera model: {camera_model}.\n"
                                f"Supported models are: {CameraModel.BC} and {CameraModel.KB}."
                            )
                    else:
                        D = np.zeros((5,), dtype=np.float64)
                        (points_2d, _) = cv2.projectPoints(object_points, rvec, tvec, K, D)

                    # points_2d: (N, 1, 2) - OpenCV format
                    annotation_pixels = points_2d.reshape(-1, 2)  # (N, 2)
                
                else:
                    raise ValueError(
                        f"Unsupported projection mode: {projection_mode}. "
                        f"Supported modes are: 'Manual' and 'OpenCV'."
                    )

                # Select annotation option based on the projection mode
                if debug is False:
                    if apply_annotation_distortion:
                        option = AnnotationOption.POLYGONS
                    else:
                        option = AnnotationOption.CUBE
                else:
                    option = AnnotationOption.POINTS

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

                if debug is False:
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
                        # Append point geo to the annotation points list
                        annotation_points_2d.append(annotation_definition.geo)
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
        if debug is False:
            for item_id in images_map.keys():
                images_map[item_id]["builder"].upload()

    @staticmethod
    def build_frame_annotations_per_frame_mapping(lidar_video_content, annotations):
        """
        Function that builds a mapping of frame annotations per frame number.
        The function also handles the case of missing inner snapshots information.
        :param lidar_video_content: DL lidar scene item content (json)
        :param annotations: DL lidar scene item annotations
        :return: mapping of frame annotations per frame number
        """
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

    def project_annotations_to_2d(self, item: dl.Item, context: dl.Context = None):
        """
        Function that projects annotations to 2D from the original lidar scene annotations.
        :param item: DL lidar scene item
        :param context: context object with custom node configuration:
        - full_annotations_only: if True, only full annotations will be projected to 2D
        - debug: if False, annotations will be uploaded to the image items, otherwise annotations will be drawn on the images locally.
        - apply_image_undistortion: if True, apply image undistortion to the images before projection
        - apply_annotation_distortion: if True, apply annotation distortion to the projected pixels
        - start_frame: if provided, only annotations from the start frame will be projected to 2D
        - end_frame: if provided, only annotations until the end frame will be projected to 2D (exclusive)
        :return: None
        """
        if context is not None and context.node is not None:
            config = context.node.metadata.get("customNodeConfig", dict())
        else:
            config = dict()
        
        start_frame = config.get("start_frame", 0)
        end_frame = config.get("end_frame", -1)  # -1 means all frames

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
        if end_frame == -1:
            end_frame = frames_count
        for frame_num in range(start_frame, end_frame):
            print("Frame number:", frame_num)

            #################
            # Handle Images #
            #################
            frame_images = lidar_video_content.get('frames', list())[frame_num].get('images', list())
            frame_annotations = frame_annotations_per_frame.get(frame_num, list())
            self.handle_frame(
                items_path=items_path,
                labels_colors=labels_colors,
                cameras_map=cameras_map,
                frame_images=frame_images,
                frame_annotations=frame_annotations,
                config=config
            )


if __name__ == "__main__":
    # frames json item ID
    item_id = '686699883d66eb96ffd891fa'
    frames_item = dl.items.get(item_id=item_id)
    # frames_item.open_in_web()

    # Create context
    context = dl.Context()
    context.node = dl.entities.node.PipelineNode(
        metadata=dict(
            customNodeConfig=dict(
            full_annotations_only=False,
            debug=True,
            apply_image_undistortion=False,
            apply_annotation_distortion=True,
            start_frame=0,
            end_frame=1,
            )
        )
    )

    runner = AnnotationProjection()
    runner.project_annotations_to_2d(
        item=frames_item,
        context=context
    )
