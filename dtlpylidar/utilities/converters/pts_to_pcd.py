import open3d as o3d
import numpy as np
from dtlpylidar.utilities.converters.base_converter import PCDConverter, DownsampleConfig


class PtsToPCD(PCDConverter):
    def __init__(self, extension: str = ".pts"):
        super().__init__(extension=extension)

    def convert_file(self, input_file: str, output_file: str = None, 
                     transform_matrix: np.ndarray = None, downsample_config: DownsampleConfig = None, **kwargs):
        """
        Convert PTS file to PCD.
        
        Args:
            input_file: Path to input PTS file
            output_file: Path to output PCD file (optional)
            transform_matrix: Transformation matrix to apply to the point cloud
            downsample_config: Downsample configuration to apply to the point cloud
        
        Returns:
            o3d.geometry.PointCloud: The converted point cloud
        """
        # Read PTS file as point cloud (Open3D supports PTS format natively)
        pcd = o3d.io.read_point_cloud(input_file)
        
        # Validate that point cloud was loaded successfully
        if not pcd.has_points():
            raise ValueError(f"PTS file '{input_file}' contains no points or could not be read")
        
        if len(pcd.points) == 0:
            raise ValueError(f"PTS file '{input_file}' is empty")
            
        pcd = self.transform_and_downsample(
            pcd=pcd, 
            output_file=output_file, 
            transform_matrix=transform_matrix,
            downsample_config=downsample_config
        )
        self.save_pcd(pcd=pcd, save_filename=output_file, check_size=False)
        return pcd
