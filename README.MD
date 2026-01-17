![BigBird](/images/Theme.png "BigBird")

# BigBird

**Note: English version of README was translated from Chinese version directly with DeepSeek. Even thought it has been reviewed manually, there still might be some mistakes.**

## Introduction

This project is designed for multi-camera RGB calibration, applicable to **any** number of cameras. It uses **graph optimization** to compute the global camera extrinsic parameters.  
The calibration pattern used is a chessboard. Currently, only a USB camera interface is provided. For other types of cameras, you need to save the detected chessboard corners as `corners.yaml`. For details, see [Chessboard Corner Data Format](#chessboard-corner-data-format).

## Methods

First, the intrinsic parameters of each camera are calculated using `OpenCV`. Then, the intrinsic parameters are fixed, and the initial extrinsic parameters are computed using `OpenCV`'s stereo calibration. Finally, graph optimization is performed.  
In the graph optimization, the vertices are the SE(3) transformations from the calibration board coordinate system to the main camera coordinate system, and the SE(3) transformations from the main camera to the secondary cameras. The edges represent the reprojection error.

## Dependencies

To simplify environment configuration, this project minimizes dependencies. The library versions used in the developer's environment are listed below. Compatibility with other versions has not been tested.  
`g2o=?, Eigen3=3.4.0, opencv=4.14.0, yaml-cpp=?`  
Here, `g2o` uses the commit from December 9, 2025.

## Usage

**Before starting, you need to set the parameters.** For parameter definitions, see #parameter-meanings.[Configure Parameter](#configure-parameter).

### Code Compilation

First, compile the project:

```bash
mkdir build
cd build
cmake ..
make
```

### Corner Detection

Next, you can use the script to start the corner detection function:

```bash
cd ..
bash scripts/detect_corners.bash
```

![Camera window](/images/image_win.png "Camera window")![Control window](/images/control_win.png "Control window")

This program visualizes the image from each camera and draws the chessboard corners. A control window will also pop up. The keyboard controls are as follows:

| Key | Function |
| --- | --- |
| `g`/`G` | Capture one frame from all cameras |
| `q`/`Q` | Exit the program without saving the detection results |
| `Esc` | Save the detection results and exit |

**Note: The control window must be focused for the controls to take effect.**

In the control window, you can see some green numbers. The numbers on the main diagonal represent how many valid frames have been captured by each camera, while the other elements indicate how many frames are simultaneously visible to two cameras.

### Graph Optimization

Run the following script to perform graph optimization:

```bash
bash scripts/solve_param.bash
```

The program will automatically compute the intrinsic and extrinsic parameters of the cameras and save them in `output/calib_result.yaml`.  
During the rough calibration stage, the program will have each secondary camera pick a frame and project the corners observed by other cameras into its own view. The colored points are the corners captured by that camera, and the green points are the projections.  
After the graph optimization is complete, the program will also visualize the optimization results for each frame, again using projection. In this visualization, the white line intersections are the detected corners, the red points are the projections under the rough extrinsic parameters, and the green points are those after graph optimization. For this window, pressing `Esc` will skip the visualization of all images.

![Rough calibrate](/images/rough_calib.png "Rough calibrate")![Result](/images/res.png "Result")

## Configure Parameter

The configs are stored in `config.yaml`.

| Parameter | Description |
| --- | --- |
| corner_out_path | Path to store detected corners |
| camera_count | Total number of cameras |
| calib_result_path | Path to store calibration results |
| corner_edge_length | Chessboard square edge length (in m) |
| corner_size | Number of chessboard corners (x, y) |
| optim_iterations | Number of graph optimization iterations |
| remove_iter | Iteration at which outliers in the edges will be removed |
| meta_data_path | Path to store optimization data records |
| image_rotation | Whether to rotate images (-1: no rotation, 0: 90° clockwise, 1: 180°, 2: 90° counterclockwise) |
| camera_params | Dictionary of startup parameters for each camera |
| delta | Breakpoint for the robust kernel function in graph optimization |
| outlier_threshold | Points with an average reprojection error greater than this value are considered outliers (in pixels) |

The camera startup parameters dictionary must contain the following keys: `id`, `width`, `height`, `fps`, `fourcc`. Here, `id` is the USB camera's system index.

## Directory Structure

```tree
.
├── CMakeLists.txt
├── config.yaml
├── images
│   ├── control_win.png
│   └── image_win.png
├── include
│   ├── Camera.hpp
│   ├── CornerData.hpp
│   └── Graph.hpp
├── output
│   ├── calib_result.yaml
│   ├── corners.yaml
│   └── meta_data.csv
├── README.MD
├── scripts
│   ├── detect_corners.bash
│   └── solve_param.bash
└── src
    ├── Camera.cpp
    ├── DetectCorner.cpp
    ├── Graph.cpp
    └── SolveParam.cpp
```

| File | Description |
| --- | --- |
| `config.yaml` | Configuration file |
| `scripts/detect_corners.bash` | Corner detection script |
| `scripts/solve_param.bash` | Graph optimization script |
| `output/corners.yaml` | Results of corner detection |
| `output/calib_result.yaml` | Final calibration results |
| `output/meta_data.csv` | Information for each iteration in graph optimization |
| `DetectCorner.cpp` | Code related to corner detection |
| `Camera.hpp`+`Camera.cpp` | Manages cameras during corner detection |
| `CornerData.hpp` | Manages corner data |
| `SolveParam.cpp` | Code related to graph optimization |
| `Graph.hpp`+`Graph.cpp` | Implementation of edges in the optimization |

## Chessboard Corner Data Format

The data is essentially a list. Each element is a dictionary representing the detection result for one image. The dictionary format is:

```python
dic{id:int, camera_id:int, corners:List[List[]]}
```

| Key | Description |
| --- | --- |
| id | Frame ID (does not need to be sequential) |
| camera_id | Camera index |
| corners | Detected corners |

A correct example of `corners.yaml` is shown below. Note that it must be placed in the `output` folder for the program to find it.

```yaml
- id: 0
  camera_id: 0
  corners:
    - [171.703308, 605.984375]
    - [167.304123, 565.771973]
    ...
    - [373.035278, 139.042175]
- id: 1
  camera_id: 0
  corners:
    - [318.778137, 521.173035]
    - [315.91626, 491.984863]
    ...
    - [457.876984, 196.106903]
...
```

## Camera Indexing

The `id` in the configuration file is the device index used by `VideoCapture` to open the camera. Internally, the program sorts these indices in ascending order and remaps them to a continuous id starting from 0. Subsequent visualizations and outputs will use this continuous id. The main camera refers to the camera with id 0.

## Result Format

Each camera will output an intrinsic parameter dictionary, for example:

```yaml
- id: 0
  intrinsic:
    - [532.58427447468182, 0, 299.59569221309891]
    - [0, 533.4434255867501, 424.98465485374732]
    - [0, 0, 1]
  dist_coeffs:
    - [0.14817218784429778, -0.29601138501316154, 0.0073656798834496684, 0.0027927246670419958, 0.20075163364797521]
```

The intrinsic parameters are defined according to the **pinhole camera model**, and the distortion model is the **radial-tangential distortion model**  (**[$k_1$, $k_2$, $p_1$, $p_2$, $k_3$]**).  
For each secondary camera, an extrinsic parameter dictionary will be output:

```yaml
- father_id: 1
  child_id: 0
  T_father_child:
    - [0.54159964267310512, -0.42394522410100416, 0.72590651878758672, -0.11655659035239203]
    - [-0.32944255076309414, 0.68740519754538576, 0.64725705877517714, -0.096402260991194005]
    - [-0.77339345278006533, -0.58969868691522198, 0.23263281335054967, 0.11489410570060896]
    - [0, 0, 0, 1]
  T_father_child_rough:
    - [0.54118900216445953, -0.42686215897784519, 0.72450200908555784, -0.11691699170810418]
    - [-0.32465224811816462, 0.68871391202361631, 0.64828548123257268, -0.097039539471639402]
    - [-0.77570315309921134, -0.58605617872171556, 0.23414263527611145, 0.11450795255146941]
    - [0, 0, 0, 1]
```

Here, `father_id` is the ID of this secondary camera, and `child_id` is the ID of the main camera. The two transformations below are both extrinsic camera parameters that transform coordinates from the main camera coordinate system to the secondary camera coordinate system. The one with the `rough` subscript is the result computed by `OpenCV` without graph optimization.
