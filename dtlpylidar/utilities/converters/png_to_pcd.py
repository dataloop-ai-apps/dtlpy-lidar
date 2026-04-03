import numpy as np
import open3d as o3d
import cv2
import pathlib
from dtlpylidar.utilities.converters.base_converter import PCDConverter, DownsampleConfig


class PngToPCD(PCDConverter):
    def __init__(self, extension: str = ".png"):
        super().__init__(extension=extension)

    def convert_file(self, input_file: str, output_file: str = None,
                     transform_matrix: np.ndarray = None, downsample_config: DownsampleConfig = None,
                     intrinsics: dict = None, depth_scale: float = 1000.0, color_file: str = None, **kwargs):
        """
        Convert a PNG file to a PCD file.
        Args:
            input_file: The path to the input PNG file.
            output_file: The path to the output PCD file.
            transform_matrix: Transformation matrix to apply to the point cloud
            downsample_config: Downsample configuration to apply to the point cloud
            intrinsics: The intrinsics of the camera.
                Example (Orthographic camera):
                {
                    'fx': 1.0,
                    'fy': 1.0,
                    'cx': 0.0,
                    'cy': 0.0,
                    'skew': 0.0,
                    'near': 0.0,
                    'far': 100.0
                }
            depth_scale: The scale of the depth image.
                Example:
                1000.0 for mm to meters
                1.0 for meters to meters
            color_file: The path to the color image file.
                Example:
                "path/to/color.png"
        Returns:
            o3d.geometry.PointCloud: The converted point cloud
        """
        if intrinsics is None:
            raise ValueError("Intrinsics are required")
        
        # Get image dimensions
        depth_img = cv2.imread(input_file, cv2.IMREAD_ANYDEPTH)
        height, width = depth_img.shape[:2]

        # Get color image
        if color_file is not None:
            color_img = cv2.imread(color_file, cv2.IMREAD_COLOR)

            if (color_img is not None) and (color_img.shape[:2] != (height, width)):
                raise ValueError("Color image dimensions do not match depth image dimensions")

            # Get colors for each point (convert BGR to RGB and normalize to 0-1)
            colors = color_img.reshape(-1, 3)
            colors = colors[:, [2, 1, 0]]  # BGR to RGB
            colors = colors.astype(np.float32) / 255.0  # Normalize to 0-1
        else:
            colors = None

        # Create coordinate grids
        x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))

        # Convert to normalized coordinates
        x_norm = (x_grid - intrinsics['cx']) / intrinsics['fx']
        y_norm = (y_grid - intrinsics['cy']) / intrinsics['fy']

        # Convert depth from millimeters to meters (assuming depth is in mm)
        depth_meters = depth_img.astype(np.float32) / depth_scale

        # Create 3D points
        x_3d = x_norm * depth_meters
        y_3d = y_norm * depth_meters
        z_3d = depth_meters

        # Stack coordinates and reshape
        points_3d = np.stack([x_3d, y_3d, z_3d], axis=-1)
        points_3d = points_3d.reshape(-1, 3) 

        # Remove invalid points (depth = 0 or NaN)
        near = intrinsics.get('near', 0.0)
        far = intrinsics.get('far', 100.0)
        valid_mask = (depth_meters.flatten() > near) & (depth_meters.flatten() < far)
        valid_points = points_3d[valid_mask]
        if len(valid_points) == 0:
            raise ValueError("No valid points found in depth image")

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(valid_points)
        if colors is not None:
            valid_colors = colors[valid_mask]
            pcd.colors = o3d.utility.Vector3dVector(valid_colors)

        pcd = self.transform_and_downsample(
            pcd=pcd, 
            output_file=output_file, 
            transform_matrix=transform_matrix,
            downsample_config=downsample_config
        )
        self.save_pcd(pcd=pcd, save_filename=output_file, check_size=False)
        return pcd
