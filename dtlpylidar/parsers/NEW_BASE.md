# Dataloop Dataset Dir
```
camera
|- camera_name_0
|  |- 0.jpg
|  |- 1.jpg
|  |- ...
|- camera_name_1
|  |- 0.jpg
|  |- 1.jpg
|  |- ...
|- ...
lidar
|- 0.pcd
|- 1.pcd
|- ...
```

# Calibration Dict Format
```
{
    "frames": {
        "0": {
            "path": <>, // for frame 0: "{cwd}/lidar/0.pcd"
            "timestamp": <>
            "position": { // lidar sensor location (used as the center of the world)
                "x": <>,
                "y": <>,
                "z": <>
            },
            "heading": { // lidar sensor rotation (Quaternion)
                "x": <>,
                "y": <>,
                "z": <>,
                "w": <>
            },
            "cameras": { // if no images are provided, add an empty dict
                "0": {
                    "path": <>, // for frame 0 camera 0: "{cwd}/cameras/{camera_name}/0.jpg" 
                    "timestamp": <>,
                    "intrinsics": { // camera intrinsic
                        "fx": <>, // Focal length in pixels.
                        "fy": <>,
                        "cx": <>, // Optical center (the principal point), in pixels.
                        "cy": <>,
                    },
                    "extrinsics": { // camera extrinsic
                        "position": { // camera location in world coordinates (in relation to the lidar sensor)
                            "x": <>,
                            "y": <>,
                            "z": <>
                        },
                        "heading": { // rotation of the camera (Quaternion)
                            "w": <>,
                            "x": <>,
                            "y": <>,
                            "z": <>
                        }
                    },
                    "distortion" : { // distortion parameters
                        "k1": <>,
                        "k2": <>,
                        "p1": <>,
                        "p2": <>,
                        "k3": <>,
                        "k4": <>
                    }
                }
            }
        }
    }
}
```

# PKL to PCD Converter:
```
import os
import pickle
import open3d as o3d
import pandas as pd


def convert_pkl_to_pcd(input_folder):
    # List all .pkl files in the input folder
    pkl_files = [f for f in os.listdir(input_folder) if f.endswith('.pkl')]

    if not pkl_files:
        print("No .pkl files found in the folder.")
        return

    # Process each .pkl file
    for pkl_file in pkl_files:
        pkl_file_path = os.path.join(input_folder, pkl_file)
        pcd_file_path = os.path.join(input_folder, pkl_file.replace('.pkl', '.pcd'))

        try:
            # Load the .pkl file
            with open(pkl_file_path, 'rb') as file:
                data = pickle.load(file)

            # Convert data to DataFrame if needed
            if isinstance(data, pd.DataFrame):
                df = data
            else:
                df = pd.DataFrame(data)

            # Check if required columns exist
            if not {'x', 'y', 'z'}.issubset(df.columns):
                print(f"Missing required columns in file: {pkl_file}")
                continue

            # Extract x, y, z coordinates
            points = df[['x', 'y', 'z']].to_numpy()

            # Create Open3D PointCloud object
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(points)

            # Save to .pcd file
            o3d.io.write_point_cloud(pcd_file_path, point_cloud)
            print(f"Successfully converted {pkl_file} to {pcd_file_path}")

        except Exception as e:
            print(f"Failed to process {pkl_file}: {e}")

if __name__ == "__main__":
    input_folder = r"..\pandaset\001\lidar"
    convert_pkl_to_pcd(input_folder)

```