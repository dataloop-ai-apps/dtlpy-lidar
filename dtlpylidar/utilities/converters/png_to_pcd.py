import numpy as np
import open3d as o3d
import cv2
import pathlib
from dtlpylidar.utilities.converters.base_converter import BaseToPCDConverter


class PngToPCD(BaseToPCDConverter):
    extension = ".png"  # Default extension

    def __init__(self, extension=None):
        super().__init__(extension=extension)

    def convert_file(self, input_file, output_file=None, 
                     intrinsics: dict = None, 
                     depth_scale: float = 1000.0, 
                     color_file = None, 
                     **kwargs):
        """
        Convert a PNG file to a PCD file.
        :param input_file: The path to the input PNG file.
        :param output_file: The path to the output PCD file.
        :param intrinsics: The intrinsics of the camera.
            Example:
            {
                'fx': 1000,
                'fy': 1000,
                'cx': 100,
                'cy': 100,
                'near': 0.0,
                'far': 100.0
            }
        :param depth_scale: The scale of the depth image.
            Example:
            1000.0 for mm to meters
            1.0 for meters to meters
        :return: The PCD file.
        """
        if intrinsics is None:
            raise ValueError("Intrinsics are required")
        
        # Get image dimensions
        depth_img = cv2.imread(str(input_file))
        height, width = depth_img.shape[:2]

        # Get color image
        if color_file is not None:
            color_img = cv2.imread(str(color_file))

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

        return pcd

    def convert_folder(self, input_folder, output_folder=None, 
                       intrinsics: dict = None, 
                       depth_scale: float = 1000.0, 
                       color_folder = None, 
                       **kwargs):
        """
        Convert a folder of PNG files to a PCD files.
        :param input_folder: The path to the input folder.
        :param output_folder: The path to the output folder.
        :param intrinsics: The intrinsics of the camera.
            Example:
            {
                'fx': 1000,
                'fy': 1000,
                'cx': 100,
                'cy': 100,
                'near': 0.0,
                'far': 100.0
            }
        :param depth_scale: The scale of the depth image.
            Example:
            1000.0 for mm to meters
            1.0 for meters to meters
        :param color_folder: The path to the color folder.
        :return: The PCD files.
        """
        sorted_flag = kwargs.get("sorted", False)
        kwargs.update({
            'intrinsics': intrinsics,
            'depth_scale': depth_scale
        })

        extension = self.extension
        if not extension.startswith("."):
            extension = f".{extension}"
        
        data_filepaths = pathlib.Path(input_folder).rglob(f"*{extension}")
        if color_folder is not None:
            data_filepaths = sorted(data_filepaths)
            color_filepaths = sorted(pathlib.Path(color_folder).rglob(f"*{extension}"))
        elif sorted_flag:
            data_filepaths = sorted(data_filepaths)
        
        output_results = []
        for idx, data_filepath in enumerate(data_filepaths):
            output_filepath = pathlib.Path(output_folder).joinpath(data_filepath.with_suffix(".pcd").relative_to(input_folder))
            if color_folder is not None:
                color_filepath = color_filepaths[idx]
                kwargs.update({
                    'color_file': color_filepath
                })
            output_result = self.convert_file(input_file=data_filepath, output_file=output_filepath, **kwargs)
            output_results.append(output_result)
        
        print(f"Successfully converted {len(output_results)} files")
        return output_results
