# Ground Detection

Coming soon.

---

## Description

Pre-processing done by dataloop on file of type `.pcd` to detect ground points in Point cloud data files.  
The detection scripts are triggered each time a file of type `.pcd` is uploaded to the platform.

---

## Ground detection algorithms

- [GndNet ground detection](https://github.com/anshulpaigwar/GndNet)

  - Requirements: GndNet model 
  - input: pcd file. 
  - output: txt file with a list of point indices that are part of ground.


- Plane segmentation algorithm:
  - input: pcd file.
  - output: txt file with a list of point indices that are part of ground.
  