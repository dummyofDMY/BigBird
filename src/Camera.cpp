#include "Camera.hpp"

void Camera::capture_and_visualize() {
        std::cout << "相机线程启动" << std::endl;
        
        // 创建OpenCV窗口
        cv::namedWindow("Camera" + std::to_string(id_) + " View", cv::WINDOW_AUTOSIZE);
        
        while (is_running_.load()) {
            // 采集图像
            cv::Mat frame;
            cap_ >> frame;

            if (frame.empty()) {
                cv::waitKey(15);
                continue;
            }
            if (rotation_flag_ == 0)
                cv::rotate(frame, frame, cv::ROTATE_90_CLOCKWISE);
            else if (rotation_flag_ == 1)
                cv::rotate(frame, frame, cv::ROTATE_180);
            else if (rotation_flag_ == 2)
                cv::rotate(frame, frame, cv::ROTATE_90_COUNTERCLOCKWISE);
            
            // // 更新当前帧（加锁保护）
            // {
            //     std::lock_guard<std::mutex> lock(frame_mutex);
            //     frame.copyTo(current_frame);
            //     new_frame_available = true;
            // }
            // cv.notify_one();  // 通知等待的线程

            // 检测图片中的棋盘格
            cv::Mat gray;
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            std::vector<cv::Point2f> corners;
            int flages = 0;
            bool need_update = corner_need_update_.load();
            if (need_update) {
                flages = cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_EXHAUSTIVE
                    + cv::CALIB_CB_ACCURACY;
            }
            bool found = cv::findChessboardCornersSB(gray, cv::Size(12, 7), corners, flages);
            if (found) {
                cv::drawChessboardCorners(frame, cv::Size(12, 7), corners, found);
                if (need_update) {
                    std::lock_guard<std::mutex> lock(corner_mutex_);
                    corners_ = corners;
                    new_corner_available_.store(true);
                    detection_done_.notify_one();  // 通知等待的线程
                }
            } else {
                if (need_update) {
                    std::lock_guard<std::mutex> lock(corner_mutex_);
                    corners_.clear();
                    new_corner_available_.store(true);
                    // std::cout << id_ << "未检测到棋盘格" << std::endl;
                    detection_done_.notify_one();  // 通知等待的线程
                }
            }
            // 缩放图像
            int max_side = std::max(frame.rows, frame.cols);
            double ratio = 320.0 / static_cast<double>(max_side);
            cv::resize(frame, frame, cv::Size(), ratio, ratio);
            
            // 显示图像
            cv::imshow("Camera" + std::to_string(id_) + " View", frame);
            
            // 等待按键（短时间等待，保持响应性）
            int key = cv::waitKey(1);
            
            // 如果用户按下ESC键，设置停止标志
            if (key == 27) {  // ESC键
                std::cout << "ESC键按下，准备停止..." << std::endl;
                stop();
                break;
            }
        }
        
        // 清理OpenCV窗口
        cv::destroyWindow("Camera View");
        std::cout << "相机线程结束" << std::endl;
    }

Camera::Camera(int id, int width, int height, int fps, std::string fourcc, int rotation_flag) :
    is_running_(true), new_corner_available_(false), corner_need_update_(false), id_(id), rotation_flag_(rotation_flag) {
        std::cout << "相机对象" << id_ << "创建" << std::endl;

        // 获取相机
        cap_ = cv::VideoCapture(id_, cv::CAP_V4L2);
        if (!cap_.isOpened()) {
            std::cerr << "无法打开相机，ID: " << id_ << std::endl;
            is_running_.store(false);
            throw std::runtime_error("无法打开相机" + std::to_string(id_));
        }

        // 设置相机参数
        int fourcc_code = cv::VideoWriter::fourcc(
            fourcc[0], fourcc[1], fourcc[2], fourcc[3]);
        cap_.set(cv::CAP_PROP_FOURCC, fourcc_code);
        cap_.set(cv::CAP_PROP_FRAME_WIDTH, width);
        cap_.set(cv::CAP_PROP_FRAME_HEIGHT, height);
        cap_.set(cv::CAP_PROP_FPS, fps);
        
        // 启动图像采集线程
        capture_thread_ = std::thread(&Camera::capture_and_visualize, this);
    }

Camera::~Camera() {
    std::cout << "相机对象" << id_ << "销毁" << std::endl;
    stop();  // 确保停止线程
}

std::vector<cv::Point2f> Camera::get_corner() {
    corner_need_update_.store(true);
    std::unique_lock<std::mutex> lock(corner_mutex_);
    // 等待新帧可用
    bool result = detection_done_.wait_for(lock, std::chrono::milliseconds(1000), [this]() {
        return new_corner_available_ || !is_running_.load(); 
    });
    corner_need_update_.store(false); 
    
    if (new_corner_available_ && is_running_.load() && result) {
        std::vector<cv::Point2f> corners;
        corners.assign(corners_.begin(), corners_.end());
        new_corner_available_ = false;
        std::cout << "相机" << id_ << "获取到 " << corners.size() << " 个角点" << std::endl;
        return corners;
    }
    std::cout << "相机" << id_ << "未找到角点" << std::endl;
    return std::vector<cv::Point2f>();  // 返回空向量
}

// cv::Mat Camera::get_visualize_frame() {
//     std::lock_guard<std::mutex> lock(image_mutex_);
//     return current_frame.clone();
// }

void Camera::stop() {
    if (is_running_.load()) {
        std::cout << "正在停止相机线程..." << std::endl;
        is_running_.store(false);
        
        // 通知所有等待的线程
        detection_done_.notify_all();
        
        // 等待线程结束
        if (capture_thread_.joinable()) {
            capture_thread_.join();
            std::cout << "相机线程已成功回收" << std::endl;
        }
    }
}
