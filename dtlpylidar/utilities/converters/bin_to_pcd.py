import struct
import numpy as np
import open3d as o3d
from dtlpylidar.utilities.converters.base_converter import BaseToPCDConverter


class BinToPCD(BaseToPCDConverter):
    extension = ".bin"  # Default extension

    def __init__(self, extension=None):
        super().__init__(extension=extension)

    def convert_file(self, input_file, output_file=None, **kwargs):
        size_float = 4
        list_pcd = []
        with open(input_file, "rb") as f:
            byte = f.read(size_float * 4)
            while byte:
                x, y, z, intensity = struct.unpack("ffff", byte)
                list_pcd.append([x, y, z])
                byte = f.read(size_float * 4)
        np_pcd = np.asarray(list_pcd)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np_pcd)
        if output_file is not None:
            o3d.io.write_point_cloud(output_file, pcd)
        return pcd
