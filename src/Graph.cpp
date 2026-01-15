#include "Graph.hpp"

// 修改 corners 的类型为 std::vector<cv::Point3f>
std::vector<cv::Point3f> MainObserveEdge::corners;
std::vector<cv::Point3f> SubObserveEdge::corners;

MainObserveEdge::MainObserveEdge(int dim,
        const cv::Mat& F,
        const cv::Mat& dist_coeffs,
        const ObserveVec& measurement) :
            F_(F), dist_coeffs_(dist_coeffs) {
    setMeasurement(measurement);
    setDimension(dim);
}

void MainObserveEdge::computeError() {
    const g2o::VertexSE3* v = static_cast<const g2o::VertexSE3*>(_vertices[0]);
    Eigen::Matrix4d board_pose = v->estimate().matrix();

    // 将位姿转换为旋转向量和平移向量
    cv::Mat R_cv(3, 3, CV_32F);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            R_cv.at<float>(i, j) = board_pose(i, j);
        }
    }
    cv::Mat rvec(3, 1, CV_32F);
    cv::Rodrigues(R_cv, rvec);
    cv::Mat tvec(3, 1, CV_32F);
    for (int i = 0; i < 3; ++i) {
        tvec.at<float>(i, 0) = board_pose(i, 3);
    }

    // // 确保 F_ 和 dist_coeffs_ 的深度一致
    // cv::Mat F_32F, dist_coeffs_32F;
    // F_.convertTo(F_32F, CV_32F);
    // dist_coeffs_.convertTo(dist_coeffs_32F, CV_32F);

    std::vector<cv::Point2f> image_points;
    cv::projectPoints(corners, rvec, tvec, F_, dist_coeffs_, image_points);

    // 计算误差
    ObserveVec image_points_eigen(image_points.size() * 2);
    for (size_t i = 0; i < image_points.size(); ++i) {
        image_points_eigen(2 * i) = image_points[i].x;
        image_points_eigen(2 * i + 1) = image_points[i].y;
    }
    _error = image_points_eigen - _measurement;
}

SubObserveEdge::SubObserveEdge(int dim,
        const cv::Mat& F,
        const cv::Mat& dist_coeffs,
        const ObserveVec& measurement) :
            F_(F), dist_coeffs_(dist_coeffs) {
    setMeasurement(measurement);
    setDimension(dim);
}

void SubObserveEdge::computeError() {
    const g2o::VertexSE3* v0 = static_cast<const g2o::VertexSE3*>(_vertices[0]);
    Eigen::Matrix4d board_pose = v0->estimate().matrix();
    const g2o::VertexSE3* v1 = static_cast<const g2o::VertexSE3*>(_vertices[1]);
    Eigen::Matrix4d T_sub_main = v1->estimate().matrix();
    Eigen::Matrix4d T = T_sub_main * board_pose;

    // 将位姿转换为旋转向量和平移向量
    cv::Mat R_cv(3, 3, CV_64F);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            R_cv.at<double>(i, j) = T(i, j);
        }
    }
    cv::Mat rvec(3, 1, CV_64F);
    cv::Rodrigues(R_cv, rvec);
    cv::Mat tvec(3, 1, CV_64F);
    for (int i = 0; i < 3; ++i) {
        tvec.at<double>(i, 0) = T(i, 3);
    }

    // 确保 F_ 和 dist_coeffs_ 的深度一致
    cv::Mat F_32F, dist_coeffs_32F;
    F_.convertTo(F_32F, CV_32F);
    dist_coeffs_.convertTo(dist_coeffs_32F, CV_32F);

    std::vector<cv::Point2f> image_points;
    cv::projectPoints(corners, rvec, tvec, F_32F, dist_coeffs_32F, image_points);

    // 计算误差
    ObserveVec image_points_eigen(image_points.size() * 2);
    for (size_t i = 0; i < image_points.size(); ++i) {
        image_points_eigen(2 * i) = image_points[i].x;
        image_points_eigen(2 * i + 1) = image_points[i].y;
    }
    _error = image_points_eigen - _measurement;
}
