import open3d as o3d
import numpy as np
from dtlpylidar.utilities.converters.base_converter import PCDConverter, DownsampleConfig


class XyzToPCD(PCDConverter):
    def __init__(self, extension: str = ".xyz"):
        super().__init__(extension=extension)

    def convert_file(self, input_file: str, output_file: str = None, 
                     transform_matrix: np.ndarray = None, downsample_config: DownsampleConfig = None, 
                     xyz_format: str = "xyz", **kwargs):
        """
        Convert XYZ file to PCD using Open3D's standard writer.
        
        XYZ format is a simple ASCII format where each line contains:
        - x y z (3 values) - basic format
        - x y z r g b (6 values) - with RGB colors
        - x y z intensity (4 values) - with intensity
        - x y z intensity r g b (7 values) - with intensity and RGB
        
        Args:
            input_file: Path to input XYZ file
            output_file: Path to output PCD file (optional)
            transform_matrix: Transformation matrix to apply to the point cloud
            downsample_config: Downsample configuration to apply to the point cloud
            xyz_format: Format of XYZ file - "xyz", "xyzrgb", "xyzi", or "xyzirgb" (default: "xyz")
                       If "auto", will detect format from first line
        Returns:
            o3d.geometry.PointCloud: The converted point cloud
        """
        points = []
        colors = []
        
        with open(input_file, 'r') as f:
            # Auto-detect format if requested
            if xyz_format == "auto":
                first_line = f.readline().strip()
                if first_line:
                    parts = first_line.split()
                    num_parts = len(parts)
                    if num_parts == 3:
                        xyz_format = "xyz"
                    elif num_parts == 4:
                        xyz_format = "xyzi"
                    elif num_parts == 6:
                        xyz_format = "xyzrgb"
                    elif num_parts == 7:
                        xyz_format = "xyzirgb"
                    else:
                        raise ValueError(f"Invalid XYZ format: points line has {num_parts} parts, expected 3, 4, 6, or 7")
                    # Reset file pointer to beginning
                    f.seek(0)
            
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                
                try:
                    parts = line.split()
                    if xyz_format == "xyz":
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        points.append([x, y, z])
                    elif xyz_format == "xyzi":
                        x, y, z, intensity = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                        points.append([x, y, z])
                        intensity_norm = intensity / 255.0
                        colors.append([intensity_norm, intensity_norm, intensity_norm])
                    elif xyz_format == "xyzrgb":
                        x, y, z, r, g, b = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                        points.append([x, y, z])
                        colors.append([r / 255.0, g / 255.0, b / 255.0])
                    elif xyz_format == "xyzirgb":
                        x, y, z, intensity, r, g, b = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])
                        points.append([x, y, z])
                        colors.append([r / 255.0, g / 255.0, b / 255.0])
                    else:
                        raise ValueError(f"Invalid XYZ format: {xyz_format}")
                except Exception as e:
                    raise ValueError(f"Error parsing XYZ file '{input_file}' line {line_num}: {line}.\nError: {e}")
        
        if len(points) == 0:
            raise ValueError(f"XYZ file '{input_file}' contains no valid points")
        
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

