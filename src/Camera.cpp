#include "Camera.hpp"

void Camera::visualize() {
    // 采集图像
    cv::Mat frame;
    cap_ >> frame;

    if (rotation_flag_ == 0)
        cv::rotate(frame, frame, cv::ROTATE_90_CLOCKWISE);
    else if (rotation_flag_ == 1)
        cv::rotate(frame, frame, cv::ROTATE_180);
    else if (rotation_flag_ == 2)
        cv::rotate(frame, frame, cv::ROTATE_90_COUNTERCLOCKWISE);

    // 检测图片中的棋盘格
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    std::vector<cv::Point2f> corners;

    bool found = cv::findChessboardCornersSB(gray, this->corner_size_, corners, 0);
    if (found) {
        cv::drawChessboardCorners(frame, this->corner_size_, corners, found);
    }

    // 缩放图像
    int max_side = std::max(frame.rows, frame.cols);
    double ratio = 320.0 / static_cast<double>(max_side);
    cv::resize(frame, frame, cv::Size(), ratio, ratio);
    
    // 显示图像
    cv::imshow("Camera" + std::to_string(id_) + " View", frame);
    
    // // 等待按键（短时间等待，保持响应性）
    // cv::waitKey(1);
}

Camera::Camera(int id, int width, int height, int fps, std::string fourcc,
        cv::Size corner_size, int rotation_flag):
        id_(id), corner_size_(corner_size), rotation_flag_(rotation_flag) {
    std::cout << "相机对象" << id_ << "创建" << std::endl;

    // 获取相机
    cap_ = cv::VideoCapture(id_, cv::CAP_V4L2);
    if (!cap_.isOpened()) {
        std::cerr << "无法打开相机，ID: " << id_ << std::endl;
        // is_running_.store(false);
        throw std::runtime_error("无法打开相机" + std::to_string(id_));
    }

    // 设置相机参数
    int fourcc_code = cv::VideoWriter::fourcc(
        fourcc[0], fourcc[1], fourcc[2], fourcc[3]);
    cap_.set(cv::CAP_PROP_FOURCC, fourcc_code);
    cap_.set(cv::CAP_PROP_FRAME_WIDTH, width);
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    cap_.set(cv::CAP_PROP_FPS, fps);
    
    // // 启动图像采集线程
    // capture_thread_ = std::thread(&Camera::capture_and_visualize, this);
    // 创建OpenCV窗口
    cv::namedWindow("Camera" + std::to_string(id_) + " View", cv::WINDOW_AUTOSIZE);
}

Camera::Camera(int id, int width, int height, int fps, std::string fourcc,
        cv::Size corner_size, int exposure, int rotation_flag):
        Camera(id, width, height, fps, fourcc, corner_size, rotation_flag) {
    cap_.set(cv::CAP_PROP_AUTO_EXPOSURE, 1); // 关闭自动曝光
    cap_.set(cv::CAP_PROP_EXPOSURE, exposure);
}

Camera::~Camera() {
    std::cout << "相机对象" << id_ << "销毁" << std::endl;
    // 清理OpenCV窗口
    cv::destroyWindow("Camera" + std::to_string(id_) + " View");
}

std::vector<cv::Point2f> Camera::get_corner() {
    cv::Mat now_image;
    bool ret = this->cap_.retrieve(now_image);
    if (!ret || now_image.empty()) {
        std::cout << "相机" << id_ << "获取图像失败，无法提取角点" << std::endl;
        return std::vector<cv::Point2f>();
    }

    if (rotation_flag_ == 0)
        cv::rotate(now_image, now_image, cv::ROTATE_90_CLOCKWISE);
    else if (rotation_flag_ == 1)
        cv::rotate(now_image, now_image, cv::ROTATE_180);
    else if (rotation_flag_ == 2)
        cv::rotate(now_image, now_image, cv::ROTATE_90_COUNTERCLOCKWISE);

    cv::Mat gray;
    cv::cvtColor(now_image, gray, cv::COLOR_BGR2GRAY);
    std::vector<cv::Point2f> corners;

    int flages = cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_EXHAUSTIVE
        + cv::CALIB_CB_ACCURACY;
    bool found = cv::findChessboardCornersSB(gray, this->corner_size_, corners, flages);
    if (!found) {
        std::cout << "相机" << id_ << "未找到角点" << std::endl;
        return std::vector<cv::Point2f>();
    }
    return corners;
}
