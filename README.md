# Dataloop Lidar SDK

<p align="center">
  <img height="40mm" src="https://dataloop.ai/wp-content/uploads/2024/02/dataloop-logo-main.svg">
</p>

---

<p align="center">
  <a href="https://dataloop.ai/solutions/lidar-annotation-platform/"> <img src="assets/dataloop_lidar_studio.png"></a>
</p>

[![versions](https://img.shields.io/pypi/pyversions/dtlpy.svg)](https://github.com/dataloop-ai/dtlpy)

---

## Installation and usage

To install the package, run the following command:

```commandline
pip install git+https://github.com/dataloop-ai-apps/dtlpy-lidar.git
```

to use the package, run the following import:

```python
import dtlpylidar
```

And now you are ready to use the package!

---

## Quick Start

After Setting up the data as described in
the [LiDAR Data setup](https://docs.dataloop.ai/docs/lidar-data-setup?highlight=lidar) documentation, you can run the
following script to create the LiDAR video file.
Run the following script to create the LiDAR video file.

```python
import dtlpy as dl
from dtlpylidar.parsers.base_parser import LidarFileMappingParser

dataset = dl.datasets.get(dataset_id="<dataset id>")
mapping_item = dataset.items.get(item_id="<mapping.json item id>")
frames_item = LidarFileMappingParser().parse_data(mapping_item=mapping_item)
frames_item.open_in_web()
```

---

## Local Visualization (using Open3D)

To visualize locally a frame from the LiDAR video file on the remote dataset, you can use the following code snippet:

```python
import dtlpy as dl
from dtlpylidar.utilities.visualizations.visualize_scene import visualize_in_open_3d

dataset = dl.datasets.get(dataset_id="<dataset id>")
frames_item = dataset.items.get(item_id="<frames.json item id>")

frame_num = 0  # frame number to visualize
dark_mode = True  # dark/light mode visualization
rgb_points_color = True  # (If the point cloud has RGB points) color the points with the RGB values

visualize_in_open_3d(frames_item=frames_item, frame_num=frame_num, dark_mode=dark_mode, 
                     rgb_points_color=rgb_points_color)
```

---

## Description

The LiDAR application includes all the pre-processing, post-processing and utilities that are needed in order to work
with lidar on Dataloop.

The application will also include utility functions to help with 3d transformations and visualizations.

Pre-processing:

* [Dataloop Base Parser](dtlpylidar/parsers/PARSERS_README.md).

## Contributors

## Contributions, Bugs and Issues - How to Contribute

We welcome anyone to help us improve this app.  
[Here's](CONTRIBUTING.md) a detailed instructions to help you open a bug or ask for a feature request
