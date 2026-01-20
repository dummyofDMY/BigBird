#pragma once

#include <opencv2/opencv.hpp>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>

class Camera {
private:
    cv::VideoCapture cap_;          // OpenCV视频捕获对象
    cv::Size corner_size_;           // 棋盘格角点尺寸
    int id_;                         // 相机ID
    int rotation_flag_;                 // 图像旋转标志
    
public:
    // 构造函数
    Camera(int id, int width, int height, int fps, std::string fourcc,
        cv::Size corner_size, int rotation_flag);
    Camera(int id, int width, int height, int fps, std::string fourcc,
        cv::Size corner_size, int exposure, int rotation_flag);
    // 析构函数
    ~Camera();
    std::vector<cv::Point2f> get_corner();
    inline bool grab() {
        return cap_.grab();
    }
    void visualize();
};