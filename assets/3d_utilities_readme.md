# 3D python utilities:

Coming soon.

---

## Description

Point cloud data visualization and transformation functions using python.

## Visualization Functions

- Visualize pcd:
  - inputs:
    - pcd local path or pcd item on dl. 
    - boolean plain uniform color or multiple colors. 

  
- Visualize ground on pcd:
  - inputs:
    - pcd local path or pcd item on dl. 
    - local path to ground map file.


- Visualize annotation on pcd:
  - inputs:
    - extrinsic: pcd extrinsic calibrations. 
    - annotation translation. 
    - annotation rotation. 
    - annotation scale. 
    - pcd local path or pcd item on dl.

## Transformation Functions 

- X-axis rotation function:
  - input: angle. 
  - output: rotation matrix on x-axis.


- Y-axis rotation function:
  - input: angle. 
  - output: rotation matrix on y-axis.
  

- Z-axis rotation function:
  - input: angle. 
  - output: rotation matrix on Z-axis.


- Rotation matrix from Euler rotation:
  - input: Euler rotation. 
  - output: rotation matrix.


- Rotation matrix from Quaternion rotation:
    - input: Quaternion rotation.
    - output: rotation matrix.


- Quaternion rotation from Euler rotation:
  - input: Euler rotation. 
  - output: Quaternion rotation.


- Euler rotation from Quaternion rotation:
  - input: Quaternion rotation. 
  - output: Euler rotation.


- Translation matrix from Translation values:
  - input: `x, y, z` translation values. 
  - output: Translation matrix.


- Translate point cloud data:
  - input:
    - cloud points: numpy.array. 
    - Translation object. 
  - output: Translated cloud points


- Rotate point cloud data:
  - input:
    - cloud points: numpy.array. 
    - Rotation object. 
  - output: rotated cloud points.