# Dtlpy Lidar Parsers

## Base Parser

### Description:

Dataloop LiDAR parser is a script that takes a dataset with raw LiDAR data and creates a LiDAR video file.
The LiDAR video file is a collection of:

* PCDs
* Image from multiple cameras
* Calibrations data

Each frame in the LiDAR video file is composed of a single instance of each one of the different sensors combined along
with their calibrations.

### Pre-requisites:

To start working with our base LiDAR parser, you need to follow the data setup as described in the following link
[LIDAR Data Setup](https://docs.dataloop.ai/docs/lidar-data-setup?highlight=lidar).

### How to run:

Once the data is set up, and the `mapping.json` file is uploaded to the platform, you can run the following script to
create the LiDAR video file.

```python
import dtlpy as dl
from dtlpylidar.parsers.base_parser import LidarFileMappingParser

dataset = dl.datasets.get(dataset_id='dataset-id')
mapping_item = dataset.items.get(item_id="<mapping.json item id>")
frames_item = LidarFileMappingParser().parse_data(mapping_item=mapping_item)
frames_item.open_in_web()
```