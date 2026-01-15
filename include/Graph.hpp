#include <g2o/types/slam3d/vertex_se3.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_binary_edge.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

using ObserveVec = Eigen::Vector<double, Eigen::Dynamic>;

class MainObserveEdge : public g2o::BaseUnaryEdge<-1, ObserveVec, g2o::VertexSE3> {
public:
    static std::vector<cv::Point3f> corners;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    MainObserveEdge(int dim,
            const cv::Mat& F,
            const cv::Mat& dist_coeffs,
            const ObserveVec& measurement);

    void computeError() override;

    void setMeasurement(const ObserveVec& m) override {
        _measurement = m;
    }

    bool read(std::istream&) override {
        return true;
    }

    bool write(std::ostream&) const override {
        return true;
    }
private:
    cv::Mat F_;  // 相机内参矩阵
    cv::Mat dist_coeffs_;  // 畸变系数
};

class SubObserveEdge : public g2o::BaseBinaryEdge<-1, ObserveVec, g2o::VertexSE3, g2o::VertexSE3> {
public:
    static std::vector<cv::Point3f> corners;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    SubObserveEdge(int dim,
            const cv::Mat& F,
            const cv::Mat& dist_coeffs,
            const ObserveVec& measurement);

    void computeError() override;

    void setMeasurement(const ObserveVec& m) override {
        _measurement = m;
    }

    bool read(std::istream&) override {
        return true;
    }

    bool write(std::ostream&) const override {
        return true;
    }
private:
    cv::Mat F_;  // 相机内参矩阵
    cv::Mat dist_coeffs_;  // 畸变系数
};