import open3d as o3d
import numpy as np
from dtlpylidar.utilities.converters.base_converter import PCDConverter, DownsampleConfig


class PlyToPCD(PCDConverter):
    def __init__(self, extension: str = ".ply"):
        super().__init__(extension=extension)

    def convert_file(self, input_file: str, output_file: str = None, 
                     transform_matrix: np.ndarray = None, downsample_config: DownsampleConfig = None,
                     mesh_sample_points: int = None, **kwargs):
        """
        Convert PLY file to PCD. Handles both point cloud PLY and mesh PLY files.
        
        Args:
            input_file: Path to input PLY file
            output_file: Path to output PCD file (optional)
            transform_matrix: Transformation matrix to apply to the point cloud
            downsample_config: Downsample configuration to apply to the point cloud
            mesh_sample_points: Number of points to sample from mesh. If None, uses instance default.
                Example:
                10000 for 10000 points
                default: vertices * 2
        Returns:
            o3d.geometry.PointCloud: The converted point cloud
        """
        # Read as mesh first - this won't crash on point cloud PLY files
        # It will return a mesh with vertices but no triangles for point cloud PLY
        ply_data = o3d.io.read_triangle_mesh(input_file)
        
        # If it has triangles, it's a mesh - convert by sampling
        if ply_data.has_triangles() and len(ply_data.triangles) > 0:
            if not ply_data.has_vertices():
                raise ValueError(f"PLY file '{input_file}' has triangles but no vertices")
            
            # Determine number of points to sample from mesh
            if mesh_sample_points is None:
                mesh_sample_points = len(ply_data.vertices) * 2
            
            # Convert mesh to point cloud by sampling points uniformly
            pcd = ply_data.sample_points_uniformly(number_of_points=mesh_sample_points)
            
            pcd = self.transform_and_downsample(
                pcd=pcd, 
                output_file=output_file, 
                transform_matrix=transform_matrix,
                downsample_config=downsample_config
            )
            self.save_pcd(pcd=pcd, save_filename=output_file, check_size=False)
            return pcd
        
        # If no triangles, it's likely a point cloud PLY - read as point cloud
        # (TriangleMesh has vertices, not points, so we need to read as PointCloud)
        else:
            # Read as point cloud to get proper PointCloud object
            pcd = o3d.io.read_point_cloud(input_file)
            
            if pcd.has_points() and len(pcd.points) > 0:
                if output_file is not None:
                    o3d.io.write_point_cloud(output_file, pcd)
                return pcd
        
        # Neither mesh with triangles nor valid point cloud
        raise ValueError(
            f"PLY file '{input_file}' is neither a valid mesh (no triangles) "
            f"nor a valid point cloud (no vertices/points)"
        )
