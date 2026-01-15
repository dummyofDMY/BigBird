#pragma once

#include <opencv2/opencv.hpp>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>

class Camera {
private:
    std::thread capture_thread_;      // 图像采集和可视化线程
    std::atomic<bool> is_running_;    // 线程运行标志
    std::mutex corner_mutex_;          // 帧数据互斥锁
    std::condition_variable detection_done_;       // 条件变量用于线程同步
    // cv::Mat current_frame;           // 当前帧图像
    // bool new_frame_available;         // 新帧可用标志
    cv::VideoCapture cap_;          // OpenCV视频捕获对象
    std::vector<cv::Point2f> corners_; // 棋盘格角点
    std::atomic<bool> new_corner_available_; // 新角点可用标志
    std::atomic<bool> corner_need_update_; // 角点需要更新标志
    int id_;                         // 相机ID
    // 图像采集和可视化线程函数
    void capture_and_visualize();
    
public:
    // 构造函数
    Camera(int id);
    // 析构函数
    ~Camera();
    // 获取当前帧（线程安全）
    std::vector<cv::Point2f> get_corner();
    // 停止相机线程
    void stop();
    // 检查是否正在运行
    bool is_running_flag() const {
        return is_running_.load();
    }
};