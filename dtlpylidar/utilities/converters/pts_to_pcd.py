import open3d as o3d
import numpy as np
from dtlpylidar.utilities.converters.base_converter import PCDConverter, DownsampleConfig


class PtsToPCD(PCDConverter):
    def __init__(self, extension: str = ".pts"):
        super().__init__(extension=extension)

    def convert_file(self, input_file: str, output_file: str = None, 
                     transform_matrix: np.ndarray = None, downsample_config: DownsampleConfig = None, 
                     pts_format: str = "xyzirgb", **kwargs):
        """
        Convert PTS file to PCD using Open3D's standard writer (packed RGB format).
        
        Args:
            input_file: Path to input PTS file
            output_file: Path to output PCD file (optional)
            transform_matrix: Transformation matrix to apply to the point cloud
            downsample_config: Downsample configuration to apply to the point cloud
            pts_format: Format of PTS file - "xyz", "xyzi", or "xyzirgb" (default: "xyzirgb")
        Returns:
            o3d.geometry.PointCloud: The converted point cloud
        """
        # Parse PTS file
        points = []
        colors = []
        
        with open(input_file, 'r') as f:
            first_line = f.readline().strip()
            try:
                int(first_line)  # Validate first line is number
            except ValueError:
                raise ValueError(f"PTS file '{input_file}' first line must be the number of points, got: '{first_line}'")
            
            for line_num, line in enumerate(f, start=2):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                
                try:
                    parts = line.split()
                    if pts_format == "xyz":
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        points.append([x, y, z])
                    elif pts_format == "xyzi":
                        x, y, z, intensity = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                        points.append([x, y, z])
                        intensity_norm = intensity / 255.0
                        colors.append([intensity_norm, intensity_norm, intensity_norm])
                    elif pts_format == "xyzirgb":
                        x, y, z, intensity, r, g, b = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])
                        points.append([x, y, z])
                        colors.append([r / 255.0, g / 255.0, b / 255.0])
                    else:
                        raise ValueError(f"Invalid PTS format: {pts_format}")
                except Exception as e:
                    raise ValueError(f"Error parsing PTS file '{input_file}' line {line_num}: {line}.\nError: {e}")
        
        if len(points) == 0:
            raise ValueError(f"PTS file '{input_file}' contains no valid points")
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points))
        if len(colors) > 0:
            pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

        pcd = self.transform_and_downsample(
            pcd=pcd, 
            output_file=output_file, 
            transform_matrix=transform_matrix,
            downsample_config=downsample_config
        )
        self.save_pcd(pcd=pcd, save_filename=output_file, check_size=False)
        return pcd
