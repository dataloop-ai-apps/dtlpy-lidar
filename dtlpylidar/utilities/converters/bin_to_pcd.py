import struct
import numpy as np
import open3d as o3d
from dtlpylidar.utilities.converters.base_converter import PCDConverter, DownsampleConfig


class BinToPCD(PCDConverter):
    def __init__(self, extension: str = ".bin"):
        super().__init__(extension=extension)

    def convert_file(self, input_file: str, output_file: str = None, 
                     transform_matrix: np.ndarray = None, downsample_config: DownsampleConfig = None,
                     bin_format: str = "xyzi", **kwargs):
        """
        Convert a BIN file to a PCD file.
        Args:
            input_file: The path to the input BIN file.
            output_file: The path to the output PCD file.
            transform_matrix: Transformation matrix to apply to the point cloud
            downsample_config: Downsample configuration to apply to the point cloud
            bin_format: Format ('xyz' or 'xyzi'). Default is 'xyzi'.
        Returns:
            o3d.geometry.PointCloud: The converted point cloud
        """
        size_float = 4
        
        # Set variables based on format
        if bin_format == 'xyzi':
            bytes_per_point = size_float * 4  # 16 bytes
            unpack_format = "ffff"  # x, y, z, intensity
        else:  # bin_format == 'xyz'
            bytes_per_point = size_float * 3  # 12 bytes
            unpack_format = "fff"  # x, y, z
        
        points = []
        colors = []
        with open(input_file, "rb") as f:
            byte = f.read(bytes_per_point)
            while byte:
                values = struct.unpack(unpack_format, byte)
                x, y, z = values[0], values[1], values[2]
                points.append([x, y, z])
                if bin_format == 'xyzi':
                    intensity = values[3]
                    colors.append([intensity / 255.0, intensity / 255.0, intensity / 255.0])
                byte = f.read(bytes_per_point)
        
        if len(points) == 0:
            raise ValueError(f"BIN file '{input_file}' contains no valid points")
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(points))
        if len(colors) > 0:
            pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors))

        pcd = self.transform_and_downsample(
            pcd=pcd, 
            output_file=output_file, 
            transform_matrix=transform_matrix,
            downsample_config=downsample_config
        )
        self.save_pcd(pcd=pcd, save_filename=output_file, check_size=False)
        return pcd
