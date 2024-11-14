# Dataloop Dataset Dir
```
cameras
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