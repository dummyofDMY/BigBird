![大鸟](/images/Theme.png "大鸟")

# BigBird

## 介绍

本工程用于多目RGB相机标定，适用于 __任意__ 多个相机，利用 __图优化__ 计算全局的相机外参。  
标定版用的是棋盘格，目前只提供了USB相机的接口，对于其他相机需要将采集到的棋盘格角点保存为`corners.yaml`才能使用，详见[棋盘格角点数据格式](#棋盘格角点数据格式)。

## 外参计算原理

先利用`OpenCV`计算出各个相机的内参，然后固定内参的的值用`OpenCV`的双目标定计算外参的初始值，然后进行图优化。  
在图优化中，顶点为标定板坐标系到主相机坐标系的SE(3)变换，以及主相机到副相机的SE(3)变换，边为重投影误差。

## 依赖

为了配置环境简单，本工程尽量减少了依赖。开发者的工程依赖的库的版本如下，未测试其它版本的兼容情况。  
`g20=?, Eigen3=3.4.0, opencv=4.14.0, yaml-cpp=?`  
其中`g2o`使用的是其在2025年12月9号的commit。

## 使用

**在开始使用前需要先设置参数**，参数定义详见[参数含义](#参数含义)

### 代码编译

首先需要编译工程

```bash
mkdir build
cd build
cmake ..
make
```

### 角点采集

接下来可以利用脚本启动角点采集功能：

```bash
cd ..
bash scripts/detect_corners.bash
```

![相机窗口](/images/image_win.png "相机窗口")![控制窗口](/images/control_win.png "控制窗口")

该程序会可视化每个相机的图片，并绘制标定板角点，同时会弹出一个控制用窗口，键盘控制设置如下：  

| 按键 | 作用 |
| --- | --- |
| `g`/`G` | 所有相机采集一帧 |
| `q`/`Q` | 退出程序而不保存采集结果 |
| `Esc` | 保存采集结果并退出 |

**注意:要聚焦于控制窗口才能生效**  

控制窗口中可以看到一些绿色的数字主对角线上的数代表改相机目前已采集到了多少有效帧，其余的元素则代表有多少帧是两相机同时可见的。

### 图优化求解

运行以下脚本以进行图优化：

```bash
bash scripts/solve_param.bash
```

程序会自动计算相机的内外参，并将其保存在`output/calib_result.yaml`中。  
在粗标定阶段程序和会让每个副相机挑以帧把别的相机下观察到的角点投影到自己的视角下。彩色的点是这个相机采集到的角点，绿色点为投影点。  
在图优化运行完后，程序还会可视化每一帧的优化结果，同样是用投影的方法可视化。其中白线焦点是检测到的角点，红色点为粗标定外参下的投影点，绿色为图优化优化过的。对于这个窗口，按下`Esc`可以跳过所有图片的可视化。

![粗标定可视化](/images/rough_calib.png "粗标定可视化")![结果可视化](/images/res.png "结果可视化")

## 参数含义

| 参数 | 说明 |
| --- | --- |
| corner_out_path | 识别角点存储位置 |
| camera_count | 相机总数 |
| calib_result_path | 标定结果存储位置 |
| corner_edge_length | 棋盘格边长 |
| corner_size | 棋盘格格点数（x, y） |
| optim_iterations | 图优化轮次 |
| remove_iter | 在这一轮优化会剔除边中的外点 |
| meta_data_path | 优化数据记录存储路径 |
| image_rotation | 是否旋转图片（-1: 不旋转, 0: 顺时针90度, 1: 180度, 2: 逆时针90度） |
| camera_params | 各个相机启动参数的字典 |
| delta | 图优化鲁棒核函数分段点 |
| outlier_threshold | 平均重投影误差大于这个值会认为是外点（单位为像素） |

其中，相机启动参数的字典需要包含以下键： `id`、`width`、`height`、`fps`、`fourcc`。这里的`id`是USB相机在系统中的编号。

## 目录结构

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

| 文件 | 说明 |
| --- | --- |
| `config.yaml` | 配置文件 |
| `scripts/detect_corners.bash` | 角点检测脚本 |
| `scripts/solve_param.bash` | 图优化脚本 |
| `output/corners.yaml` | 角点检测的结果 |
| `output/calib_result.yaml` | 标定的最终结果 |
| `output/meta_data.csv` | 图优化中每一轮的信息 |
| `DetectCorner.cpp` | 角点检测相关代码 |
| `Camera.hpp`+`Camera.cpp` | 在角点检测中管理相机 |
| `CornerData.hpp` | 管理角点数据 |
| `SolveParam.cpp` | 图优化相关代码 |
| `Graph.hpp`+`Graph.cpp` | 优化中边的实现 |

## 棋盘格角点数据格式

实际为一个字典，字典格式为：

```python
dic{id:int, camera_id:int, corners:List[List[]]}
```

| 键 | 说明 |
| --- | --- |
| id | 帧id，不要求连续 |
| camera_id | 相机序号 |
| corners | 识别到的角点 |

## 相机编号

配置文件中的`id`是`VideoCapture`打开设备的编号，程序内部会将这些编号升序排列再编成从0开始的连续的id，后续可视化、输出的id都是这个连续的编号。同时，主相机指的是id为0的相机。
