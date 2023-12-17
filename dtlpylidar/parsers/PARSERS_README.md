# Dtlpy Lidar Parsers

## Base Parser

### Description:

A service that parses lidar data inside a Dataloop Dataset,
and stitches all the files together to create a LiDAR video file.

The service support datasets with files structure as described in:
[LIDAR Data Setup](https://dataloop.ai/docs/lidar-data-setup).


### How to run:

To run the service on a lidar dataset do as follows:
1. Make sure the service is deployed on the requested project.
2. Go to the requested lidar dataset and execute the service on the `mapping.json` item 
   (See example of supported [mapping.json](assets%2Fmapping.json) file).
3. Wait until the service finishes the execution and a `frames.json` item gets created.

### Flow:

The service works as follows:
1. Receives as an input `mapping.json` item and opens it.
2. Initializes a `LidarScene` object for holding the parsed lidar data.
3. Goes over each available frame in the `mapping.json` item, and does as follows:
   1. Gets the frame PCD file and it's calibrations data.
   2. Gets the frame Camera Images and their calibrations data.
   3. Stitches the frame files and append them to the `LidarScene` object.
4. Builds from the `LidarScene` object the `frames.json` item (LiDAR video file) and upload it to the dataset.

### Requirements:

`dtlpy`
