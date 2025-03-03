# Dtlpy LiDAR Parsers

## [Base Parser](base_parser.py)

### Description:

Dataloop LiDAR parser is a script that takes a dataset with raw LiDAR data and creates a LiDAR video file.
The LiDAR video file is a collection of:

* PCDs
* Image from multiple camera sources
* Calibrations data

Each frame in the LiDAR video file is composed of a single instance of each one of the different sensors combined along
with their calibrations.

### Pre-requisites:

To start working with our base LiDAR parser, you need to follow the data setup as described in the following link
[LiDAR Data Setup](https://docs.dataloop.ai/docs/lidar-data-setup).

### How to run:

Once all files are ready, and the `mapping.json` file is uploaded to the platform, to create the LiDAR video file (of all the PCD files stitched together), run the following script:

```python
import dtlpy as dl
from dtlpylidar.parsers.base_parser import LidarFileMappingParser

dataset = dl.datasets.get(dataset_id='dataset-id')
mapping_item = dataset.items.get(item_id="<mapping.json item id>")
frames_item = LidarFileMappingParser().parse_data(mapping_item=mapping_item)
frames_item.open_in_web()
```


## [Custom Base Parser](custom_base_parser.py)

### Description:

Custom LiDAR parser is a script that set up a LiDAR scene dataset on the Dataloop platform for any directory structure of a LiDAR scene.

See [Using Custom LiDAR Parser](https://developers.dataloop.ai/tutorials/data_management/items_and_annotations/other_data_types/lidar/chapter#using-custom-lidar-parser)
for more information.
