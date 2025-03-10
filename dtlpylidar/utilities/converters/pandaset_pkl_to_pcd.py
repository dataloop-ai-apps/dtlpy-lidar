import os
import pathlib
import json
import pickle
import open3d as o3d
import pandas as pd

from dtlpylidar.utilities import transformations as transformations


def convert_pkl_to_pcd(input_folder, output_folder, apply_transformation=True):
    if apply_transformation is True:
        poses_filepath = os.path.join(input_folder, 'poses.json')
        with open(poses_filepath, 'rb') as fp:
            poses_data = json.load(fp=fp)

    # List all .pkl files in the input folder
    pkl_filepaths = sorted(pathlib.Path(input_folder).rglob('*.pkl'))

    if len(pkl_filepaths) == 0:
        print("No .pkl files found in the folder.")
        return

    # Process each .pkl file
    for idx, pkl_filepath in enumerate(pkl_filepaths):
        if apply_transformation is True:
            pose_data = poses_data[idx]
        else:
            pose_data = None
        pcd_filepath = os.path.join(
            output_folder,
            pkl_filepath.with_suffix(".pcd").relative_to(input_folder)
        )
        try:
            # Load the .pkl file
            with open(pkl_filepath, 'rb') as file:
                data = pickle.load(file)

            # Convert data to DataFrame if needed
            if isinstance(data, pd.DataFrame):
                df = data
            else:
                df = pd.DataFrame(data)

            # Check if required columns exist
            if not {'x', 'y', 'z'}.issubset(df.columns):
                print(f"Missing required columns in file: {pkl_filepath}")
                continue

            # Extract x, y, z coordinates
            points = df[['x', 'y', 'z']].to_numpy()

            # Create Open3D PointCloud object
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(points)

            if pose_data is not None:
                # Apply transformation
                position = [
                    pose_data.get("position", dict()).get("x", 0),
                    pose_data.get("position", dict()).get("y", 0),
                    pose_data.get("position", dict()).get("z", 0)
                ]
                heading = [
                    pose_data.get("heading", dict()).get("x", 0),
                    pose_data.get("heading", dict()).get("y", 0),
                    pose_data.get("heading", dict()).get("z", 0),
                    pose_data.get("heading", dict()).get("w", 0)
                ]
                rotation = transformations.rotation_matrix_from_quaternion(*heading)
                transform = transformations.calc_transform_matrix(rotation=rotation, position=position)
                point_cloud.transform(transform)

            # Save to .pcd file
            o3d.io.write_point_cloud(str(pcd_filepath), point_cloud)
            print(f"Successfully converted '{pkl_filepath}' to '{pcd_filepath}'")

        except Exception as e:
            print(f"Failed to process {pkl_filepath}: {e}")


def test_convert_pkl_to_pcd():
    input_folder = r"pandaset/001/lidar"
    output_folder = r"pandaset/001/lidar"
    apply_transformation = True
    convert_pkl_to_pcd(input_folder=input_folder, output_folder=output_folder, apply_transformation=apply_transformation)


if __name__ == "__main__":
    test_convert_pkl_to_pcd()
