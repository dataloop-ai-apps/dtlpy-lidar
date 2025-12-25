from abc import ABC
import os
import open3d as o3d
import numpy as np

class DownsampleConfig(ABC):
    def __init__(self, target_size_mb: float = 70):
        self.target_size_mb = target_size_mb


class UniformDownsampleConfig(DownsampleConfig):
    def __init__(self, target_size_mb: float = 70, initial_every_k: int = 2, every_k_step: int = 1):
        super().__init__(target_size_mb=target_size_mb)
        self.initial_every_k = initial_every_k
        self.every_k_step = every_k_step


class VoxelDownsampleConfig(DownsampleConfig):
    def __init__(self, target_size_mb: float = 70, initial_voxel_size: float = 0.45, voxel_size_step: float = 0.1):
        super().__init__(target_size_mb=target_size_mb)
        self.initial_voxel_size = initial_voxel_size
        self.voxel_size_step = voxel_size_step


class PCDConverter:
    def __init__(self, extension: str = ".pcd"):
        self.extension = extension

    ########
    # Save #
    ########
    @staticmethod
    def save_pcd(pcd: o3d.geometry.PointCloud, save_filename: str = None, check_size: bool = True):
        if save_filename is not None:
            o3d.io.write_point_cloud(save_filename, pcd, write_ascii=True)
            if check_size:
                size_mb = os.path.getsize(save_filename) / (1024 * 1024)
                return size_mb

    ##############
    # Downsample #
    ##############
    @staticmethod
    def pcd_uniform_downsample(pcd: o3d.geometry.PointCloud, output_file: str, downsample_config: UniformDownsampleConfig):
        # Load original compressed point cloud
        print(f"Original number of points: {len(pcd.points)}")

        # Start downsampling loop
        current_every_k = downsample_config.initial_every_k
        while True:
            # Downsample uniformly: keep 1 out of every 'k' points
            pcd_down = pcd.uniform_down_sample(every_k_points=current_every_k)

            # Save as ASCII
            size_mb = PCDConverter.save_pcd(pcd=pcd_down, save_filename=output_file, check_size=True)
            print(f"every_k_points: {current_every_k} -> File size: {size_mb:.2f} MB")

            # Check size
            if size_mb <= downsample_config.target_size_mb:
                print("Target size reached!")
                break
            else:
                # More aggressive downsampling: keep fewer points
                current_every_k += downsample_config.every_k_step  # Increase by 1 each iteration

        return pcd_down

    @staticmethod
    def pcd_voxel_downsample(pcd: o3d.geometry.PointCloud, output_file: str, downsample_config: VoxelDownsampleConfig):
        # Load original compressed point cloud
        print(f"Original number of points: {len(pcd.points)}")

        # Start downsampling loop
        current_voxel_size = downsample_config.initial_voxel_size
        while True:
            # Downsample
            pcd_down = pcd.voxel_down_sample(voxel_size=current_voxel_size)

            # Save as ASCII
            size_mb = PCDConverter.save_pcd(pcd=pcd_down, save_filename=output_file, check_size=True)
            print(f"Voxel size: {current_voxel_size:.4f} -> File size: {size_mb:.2f} MB")

            # Check size
            if size_mb <= downsample_config.target_size_mb:
                print("Target size reached!")
                break
            else:
                # Increase voxel size to downsample more aggressively
                current_voxel_size += downsample_config.voxel_size_step

        return pcd_down

    ###########
    # Convert #
    ###########
    def transform_and_downsample(self, pcd: o3d.geometry.PointCloud, output_file: str = None, 
                                 transform_matrix: np.ndarray = None, downsample_config: DownsampleConfig = None, **kwargs):
        if transform_matrix is not None:
            pcd.transform(transform_matrix)
        
        if downsample_config is not None:
            if isinstance(downsample_config, UniformDownsampleConfig):
                pcd = self.pcd_uniform_downsample(
                    pcd=pcd, 
                    output_file=output_file, 
                    downsample_config=downsample_config
                )
            elif isinstance(downsample_config, VoxelDownsampleConfig):
                pcd = self.pcd_voxel_downsample(
                    pcd=pcd, 
                    output_file=output_file, 
                    downsample_config=downsample_config
                )
            else:
                raise ValueError(f"Invalid downsample config: {downsample_config}")
        
        return pcd

    # NOTE: Useful to convert non-ASCII pcd files to ASCII
    def convert_file(self, input_file: str, output_file: str = None, 
                     transform_matrix: np.ndarray = None, downsample_config: DownsampleConfig = None, **kwargs):
        """
        Convert a file to a PCD file. This method is used to convert a single file to a PCD file.
        Args:
            input_file: The path to the input file.
            output_file: The path to the output PCD file.
            transform_matrix: Transformation matrix to apply to the point cloud
            downsample_config: Downsample configuration to apply to the point cloud
        Returns:
            o3d.geometry.PointCloud: The converted point cloud
        """
        pcd = o3d.io.read_point_cloud(input_file)
        pcd = self.transform_and_downsample(
            pcd=pcd, 
            output_file=output_file, 
            transform_matrix=transform_matrix,
            downsample_config=downsample_config
        )
        self.save_pcd(pcd=pcd, save_filename=output_file, check_size=False)
        return pcd


    def convert_files(self, input_file_list: list[str], output_file_list: list[str], 
                      kwargs_list: list[dict] = None, **kwargs):
        """
        Convert a list of files to a list of PCD files. This method is used to convert a list of files to a list of PCD files.
        Args:
            input_file_list: The list of paths to the input files.
            output_file_list: The list of paths to the output PCD files.
                Example:
                [
                    'path/to/file1_pcd.pcd',
                    'path/to/file2_pcd.pcd',
                ]
            kwargs_list: The list of kwargs for each file.
                Example:
                [
                    {
                        'transform_matrix': np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
                        'downsample_config': UniformDownsampleConfig(target_size_mb=70, initial_every_k=2, every_k_step=1),
                    },
                ]
        Returns:
            The list of converted point clouds.
                Example:
                [
                    {
                        'input_file': 'path/to/file1.pcd',
                        'output_file': 'path/to/file1_pcd.pcd',
                    },
                ]
        """
        if kwargs_list is None:
            kwargs_list = [{}] * len(input_file_list)

        if len(input_file_list) != len(output_file_list) or len(input_file_list) != len(kwargs_list):
            raise ValueError(
                f"Lengths not matching: "
                f" - input files ({len(input_file_list)}), "
                f" - output files ({len(output_file_list)}), "
                f" - kwargs list ({len(kwargs_list)}) must have the same length"
            )

        output_results = []
        for input_file, output_file, kwargs_item in zip(input_file_list, output_file_list, kwargs_list):
            output_result = self.convert_file(input_file=str(input_file), output_file=str(output_file), **kwargs_item)
            output_results.append(output_result)
        
        print(f"Successfully converted {len(output_results)} files")
        return output_results


def _tests():
    pcd_converter = PCDConverter()

    # Uniform downsampling settings
    input_file = r'./5080_54435.pcd'
    output_file = r'./5080_54435_ascii_uniform.pcd'
    target_size_mb = 70
    initial_every_k = 2  # Start by keeping every 2nd point
    every_k_step = 1  # Increase by 1 each iteration
    pcd_converter.pcd_uniform_downsample(
        input_file=input_file, 
        output_file=output_file, 
        target_size_mb=target_size_mb, 
        initial_every_k=initial_every_k, 
        every_k_step=every_k_step
    )

    # Voxel downsampling settings
    input_file = r'./5080_54435.pcd'
    output_file = r'./5080_54435_ascii_voxel.pcd'
    target_size_mb = 70
    initial_voxel_size = 0.45  # Start with small voxel size (higher quality)
    voxel_size_step = 0.1  # Increase voxel size by 10 cm each iteration
    pcd_converter.pcd_voxel_downsample(
        input_file=input_file, 
        output_file=output_file, 
        target_size_mb=target_size_mb, 
        initial_voxel_size=initial_voxel_size, 
        voxel_size_step=voxel_size_step
    )


if __name__ == "__main__":
    _tests()
