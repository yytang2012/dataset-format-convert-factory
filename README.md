# dataset-format-convert-factory

### Kitti
Description

reference [Link](https://github.com/bostondiditeam/kitti/blob/master/resources/devkit_object/readme.txt)


| Values  | Name         |    Description| 
| ---------|--------------| ---------------| 
|    1   | type         |     Describes the type of object: 'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',                      'Misc' or 'DontCare'| 
| 1    | truncated    |  Float from 0 (non-truncated) to 1 (truncated), where truncated refers to the object leaving image boundaries|
|    1  | occluded     |     Integer (0,1,2,3) indicating occlusion state: 0 = fully visible, 1 = partly occluded 2 = largely occluded, 3 = unknown|
|    1  | alpha        |  Observation angle of object, ranging [-pi..pi]|
|    4  | bbox         |     2D bounding box of object in the image (0-based index): contains left, top, right, bottom pixel coordinates|
|    3  | dimensions   |   3D object dimensions: height, width, length (in meters)|
|    3  |   location   |   3D object location x,y,z in camera coordinates (in meters)|
|    1  |   rotation_y |   Rotation ry around Y-axis in camera coordinates [-pi..pi]|
|    1  |   score      |   Only for results: Float, indicating confidence in detection, needed for p/r curves, higher is better.|