#include <string>
#include <vector>
#include <iostream>
#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include <g2o/types/slam3d/vertex_se3.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

#include "CornerData.hpp"
#include "Graph.hpp"

struct InternalParam {
    int id;
    Eigen::Matrix3d intrinsic;
    Eigen::Vector<double, 5> dist_coeffs;
};

struct ExternalParam {
    int father_id;
    int child_id;
    Eigen::Matrix4d T;
};

struct IterationMetadata {
    int iteration;
    double chi2;
    double lambda;
    int levenbergIter;
    double average_error;
};

std::vector<cv::Point2f> project_points(
        const std::vector<cv::Point3f>& object_points,
        const Eigen::Matrix4d& T,
        const Eigen::Matrix3d& intrinsic,
        const Eigen::Vector<double, 5>& dist_coeffs) {
    // 转换为向量形式
    cv::Mat rvec(3, 1, CV_32F), tvec(3, 1, CV_32F);
    Eigen::Matrix3d R_eigen = T.block<3,3>(0,0);
    cv::Mat R_cv(3, 3, CV_64F);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            R_cv.at<double>(i, j) = R_eigen(i, j);
        }
    }
    cv::Rodrigues(R_cv, rvec);
    for (int i = 0; i < 3; ++i) {
        tvec.at<float>(i, 0) = static_cast<float>(T(i, 3));
    }
    cv::Mat camera_matrix(3, 3, CV_64F);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            camera_matrix.at<double>(i, j) = intrinsic(i, j);
        }
    }
    cv::Mat dist_coeffs_cv(5, 1, CV_64F);
    for (int i = 0; i < 5; ++i) {
        dist_coeffs_cv.at<double>(i, 0) = dist_coeffs(i);
    }
    std::vector<cv::Point2f> image_points;
    cv::projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs_cv, image_points);
    return image_points;
}

void visualize_graph_optimization_corners(
        const std::vector<DetectCorner>& detected_corners,
        const std::vector<InternalParam>& internal_params,
        const std::vector<ExternalParam>& external_params,
        const std::vector<ExternalParam>& rough_external_params,
        const std::vector<cv::Point3f>& object_points,
        const std::map<int, Eigen::Matrix4d>& board_poses,
        const std::map<int, Eigen::Matrix4d>& rough_board_poses,
        const std::vector<std::pair<int, int>>& outlier_edges,
        cv::Size image_size,
        cv::Size corner_size,
        int camera_count) {
    // 遍历每一帧，可视化重投影结果
    std::map<int, std::map<int, std::vector<cv::Point2f>>> all_corners;
    for (const auto& dc : detected_corners) {
        all_corners[dc.id][dc.camera_id] = dc.corners;
    }

    for (const auto& board_pair : board_poses) {
        int frame_id = board_pair.first;
        const auto& cameras_corners = all_corners.at(frame_id);
        for (const auto& cam_pair : cameras_corners) {
            int cam_id = cam_pair.first;
            const auto& img_corners = cam_pair.second;

            cv::Mat control_image(image_size, CV_8UC3, cv::Scalar(50, 50, 50));
            // cv::drawChessboardCorners(control_image, corner_size, img_corners, true);
            // 同行列相邻点连线
            for (int r = 0; r < corner_size.height; ++r) {
                for (int c = 0; c < corner_size.width - 1; ++c) {
                    cv::line(control_image, img_corners[r * corner_size.width + c], img_corners[r * corner_size.width + c + 1], cv::Scalar(255, 255, 255), 1);
                }
            }
            // 同行列相邻点连线
            for (int c = 0; c < corner_size.width; ++c) {
                for (int r = 0; r < corner_size.height - 1; ++r) {
                    cv::line(control_image, img_corners[r * corner_size.width + c], img_corners[(r + 1) * corner_size.width + c], cv::Scalar(255, 255, 255), 1);
                }
            }
            
            // 查找内参
            Eigen::Matrix3d intrinsic_mat;
            Eigen::Vector<double, 5> dist_coeffs;
            for (const auto& iparam : internal_params) {
                if (iparam.id == cam_id) {
                    intrinsic_mat = iparam.intrinsic;
                    dist_coeffs = iparam.dist_coeffs;
                    break;
                }
            }

            // 查找粗略外参
            Eigen::Matrix4d T_cam_rough;
            if (cam_id == 0) {
                T_cam_rough = Eigen::Matrix4d::Identity();
            } else {
                for (const auto& eparam : rough_external_params) {
                    if (eparam.father_id == cam_id) {
                        T_cam_rough = eparam.T;
                        break;
                    }
                }
            }
            Eigen::Matrix4d T_board_rough = rough_board_poses.at(frame_id);
            Eigen::Matrix4d T_rough = T_cam_rough * T_board_rough;
            std::vector<cv::Point2f> projected_points_rough = project_points(object_points, T_rough, intrinsic_mat, dist_coeffs);

            for (size_t i = 0; i < projected_points_rough.size(); ++i) {
                cv::circle(control_image, projected_points_rough[i], 2, cv::Scalar(0, 0, 255), -1);
            }

            // 查找外参
            Eigen::Matrix4d T_cam;
            if (cam_id == 0) {
                T_cam = Eigen::Matrix4d::Identity();
            } else {
                for (const auto& eparam : external_params) {
                    if (eparam.father_id == cam_id) {
                        T_cam = eparam.T;
                        break;
                    }
                }
            }
            // 查找标定板位姿
            Eigen::Matrix4d T_board = board_pair.second;
            Eigen::Matrix4d T = T_cam * T_board;
            std::vector<cv::Point2f> projected_points = project_points(object_points, T, intrinsic_mat, dist_coeffs);

            for (size_t i = 0; i < projected_points.size(); ++i) {
                cv::circle(control_image, projected_points[i], 1, cv::Scalar(0, 255, 0), -1);
            }

            // 标注帧序号和相机序号
            std::string text = "Frame " + std::to_string(frame_id) + ", Camera " + std::to_string(cam_id);
            cv::Scalar text_color = cv::Scalar(255, 255, 255);
            if (std::find(outlier_edges.begin(), outlier_edges.end(), std::make_pair(frame_id, cam_id)) != outlier_edges.end()) {
                text += " (Outlier)";
                text_color = cv::Scalar(0, 0, 255);
            }
            cv::putText(control_image, text, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2);
            // 标注图例
            cv::putText(control_image, "Green: Optimized Projection", cv::Point(10, image_size.height - 60),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
            cv::putText(control_image, "Red: Rough Projection", cv::Point(10, image_size.height - 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
            // 提示按键
            cv::putText(control_image, "Press ESC to stop visualization, any other key to continue", cv::Point(10, image_size.height - 90),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);
            cv::imshow("Graph Optimization Visualization", control_image);
            int key = cv::waitKey(0);
            if (key == 27) {
                return;
            }
        }
    }
}

void rough_calib(const std::vector<DetectCorner>& detected_corners,
        std::vector<InternalParam>& internal_params,
        std::vector<ExternalParam>& external_params,
        std::map<int, Eigen::Matrix4d>& board_poses,
        const std::vector<cv::Point3f>& object_points,
        cv::Size camera_size,
        int camera_count) {
    std::vector<std::map<int, std::vector<cv::Point2f>>> all_corners(camera_count);
    for (const auto& dc : detected_corners) {
        all_corners[dc.camera_id][dc.id] = dc.corners;
    }
    // 对每个相机进行内参标定
    std::map<int, cv::Mat> camera_matrices;
    std::map<int, cv::Mat> dist_coeffs_map;
    std::map<int, std::map<int, Eigen::Matrix4d>> board_poses_cameras;  // frame_id -> (相机id -> 4x4位姿矩阵)
    for (int cam_id = 0; cam_id < camera_count; ++cam_id) {
        std::vector<std::vector<cv::Point3f>> obj_pts_vec;
        std::vector<std::vector<cv::Point2f>> img_pts_vec;
        std::vector<int> used_frame_ids;
        for (const auto& pair : all_corners[cam_id]) {
            obj_pts_vec.push_back(std::vector<cv::Point3f>(object_points.begin(), object_points.end()));
            img_pts_vec.push_back(pair.second);
            used_frame_ids.push_back(pair.first);
        }

        cv::Mat camera_matrix = cv::Mat::eye(3, 3, CV_64F);
        cv::Mat dist_coeffs = cv::Mat::zeros(5, 1, CV_64F);
        std::vector<cv::Mat> rvecs, tvecs;
        cv::TermCriteria term_crit(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 100, DBL_EPSILON);
        double rms = cv::calibrateCamera(obj_pts_vec, img_pts_vec, camera_size,
                camera_matrix, dist_coeffs, rvecs, tvecs,
                0, term_crit);
        std::cout << "Camera " << cam_id << " calibrated with RMS error = " << rms << std::endl;
        
        // 保存标定板位姿
        for (size_t i = 0; i < used_frame_ids.size(); ++i) {
            int frame_id = used_frame_ids[i];
            cv::Mat R;
            cv::Rodrigues(rvecs[i], R);
            cv::Mat T = tvecs[i];
            Eigen::Matrix4d T_eigen = Eigen::Matrix4d::Identity();
            for (int r = 0; r < 3; ++r) {
                T_eigen(r, 3) = T.at<double>(r, 0);
                for (int c = 0; c < 3; ++c) {
                    T_eigen(r, c) = R.at<double>(r, c);
                }
            }
            board_poses_cameras[frame_id][cam_id] = T_eigen;
        }


        InternalParam iparam;
        iparam.id = cam_id;
        iparam.dist_coeffs = Eigen::Map<Eigen::Vector<double, 5>>(dist_coeffs.ptr<double>());
        // 矩阵要逐元素赋值，避免列存储和行存储的问题
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                iparam.intrinsic(i, j) = camera_matrix.at<double>(i, j);
            }
        }
        internal_params.push_back(iparam);
        camera_matrices[cam_id] = camera_matrix;
        dist_coeffs_map[cam_id] = dist_coeffs;
    }

    // 进行外参标定
    // 是以其他相机为基准，相机0系到其他相机系的变换
    cv::Mat cam1_matrix = camera_matrices[0];  // 相机0为“第一个相机”
    cv::Mat dist1_coeffs = dist_coeffs_map[0];  // 官方文档说得到的是第一个相机到第二个相机的变换
    for (int cam_id = 1; cam_id < camera_count; ++cam_id) {
        cv::Mat cam2_matrix = camera_matrices[cam_id];
        cv::Mat dist2_coeffs = dist_coeffs_map[cam_id];

        std::vector<std::vector<cv::Point3f>> obj_pts_vec;
        std::vector<std::vector<cv::Point2f>> img_pts_vec1, img_pts_vec2;
        for (const auto& pair : all_corners[cam_id]) {
            int frame_id = pair.first;
            try {
                const auto& corners2 = all_corners[0].at(frame_id);
                obj_pts_vec.push_back(std::vector<cv::Point3f>(object_points.begin(), object_points.end()));
                img_pts_vec1.push_back(corners2);
                img_pts_vec2.push_back(pair.second);
            } catch (const std::out_of_range&) {
                // frame_id在cam_id中存在，但在cam2中不存在，跳过
                continue;
            }
        }

        if (obj_pts_vec.size() < 1) {
            std::cerr << "Camera 0 has no common frames with Camera " << cam_id << " for extrinsic calibration." << std::endl;
            throw std::runtime_error("Insufficient common frames for extrinsic calibration.");
        } else {
            std::cout << "Camera 0 and Camera " << cam_id << " have " << obj_pts_vec.size() << " common frames for extrinsic calibration." << std::endl;
        }

        cv::Mat R, T, E, F;
        double rms = cv::stereoCalibrate(
            obj_pts_vec, img_pts_vec1, img_pts_vec2,
            cam1_matrix, dist1_coeffs,
            cam2_matrix, dist2_coeffs,
            camera_size,
            R, T, E, F,
            cv::CALIB_FIX_INTRINSIC,
            cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-6)
        );
        std::cout << "Camera 0 to Camera " << cam_id << " extrinsic calibrated with RMS error = " << rms << std::endl;
        ExternalParam eparam;
        eparam.father_id = cam_id;
        eparam.child_id = 0;
        // 构造4x4变换矩阵
        eparam.T = Eigen::Matrix4d::Identity();
        // R是3x3旋转矩阵，T是3x1平移
        for (int i = 0; i < 3; ++i) {
            eparam.T(i, 3) = T.at<double>(i, 0);
            for (int j = 0; j < 3; ++j) {
                eparam.T(i, j) = R.at<double>(i, j);
            }
        }
        // eparam.T = eparam.T.inverse().eval();
        external_params.push_back(eparam);

        
    }

    // 挑一帧把点重投影到camX系检查
    std::map<int, std::map<int, std::vector<cv::Point2f>>> all_corners_vis;
    for (const auto& dc : detected_corners) {
        all_corners_vis[dc.id][dc.camera_id] = dc.corners;
    }
    for (int cam_id = 1; cam_id < camera_count; ++cam_id) {
        int check_frame_id = -1;
        int other_cam_id = 1;
        for (const auto& pair : all_corners_vis) {
            const auto& fram_corners = pair.second;
            if (fram_corners.size() > 1) {
                try {
                    const auto& corners = fram_corners.at(cam_id);
                    check_frame_id = pair.first;
                    for (const auto& camera_corner : fram_corners) {
                        if (camera_corner.first != cam_id) {
                            other_cam_id = camera_corner.first;
                            break;
                        }
                    }
                    throw std::out_of_range("Uncommon frame found.");
                } catch (const std::out_of_range&) {
                    continue;
                }
            }
        }
        if (check_frame_id == -1) {
            throw std::runtime_error("No valid frame found for extrinsic calibration check.");
        }
        Eigen::Matrix4d T_board = board_poses_cameras.at(check_frame_id).at(other_cam_id);
        Eigen::Matrix4d T_cam_me;
        for (const auto& eparam : external_params) {
            if (eparam.father_id == cam_id) {
                T_cam_me = eparam.T;
                break;
            }
        }
        Eigen::Matrix4d T_cam_other;
        if (other_cam_id == 0) {
            T_cam_other = Eigen::Matrix4d::Identity();
        } else {
            for (const auto& eparam : external_params) {
                if (eparam.father_id == other_cam_id) {
                    T_cam_other = eparam.T;
                    break;
                }
            }
        }
        Eigen::Matrix4d T_vis = T_cam_me * T_cam_other.inverse().eval() * T_board;
        std::vector<cv::Point2f> projected_points = project_points(object_points, T_vis, internal_params[cam_id].intrinsic, internal_params[cam_id].dist_coeffs);
        cv::Mat vis_image(camera_size, CV_8UC3, cv::Scalar(50, 50, 50));
        // 绘制检测到的角点
        const auto& detected_corners_vec = all_corners_vis.at(check_frame_id).at(cam_id);
        cv::drawChessboardCorners(vis_image, cv::Size(12, 7), detected_corners_vec, true);
        // 绘制重投影点
        for (size_t i = 0; i < projected_points.size(); ++i) {
            cv::circle(vis_image, projected_points[i], 2, cv::Scalar(0, 255, 0), 2);
        }
        
        // 标注图像信息
        std::string text = "Camera " + std::to_string(other_cam_id) + " to Camera " + std::to_string(cam_id);
        cv::putText(vis_image, text, cv::Point(10, 30),
            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
        cv::imshow("Rough calib", vis_image);
        cv::waitKey(0);
    }
    cv::destroyAllWindows();
        

    // 为每一帧计算一个初始的标定板位姿（相对于相机0系）
    for (const auto& pair : board_poses_cameras) {
        int frame_id = pair.first;
        const std::map<int, Eigen::Matrix4d>& cam_poses = pair.second;
        try {
            Eigen::Matrix4d T_cam0 = cam_poses.at(0);  // 相机0系下的标定板位姿
            board_poses[frame_id] = T_cam0;
        } catch (const std::out_of_range&) {
            int camera_id = cam_poses.begin()->first;
            Eigen::Matrix4d T_camX = cam_poses.begin()->second;
            // 找到相机X系到相机0系的变换
            Eigen::Matrix4d T_camX_cam0;
            for (const auto& eparam : external_params) {
                if (eparam.father_id == camera_id && eparam.child_id == 0) {
                    T_camX_cam0 = eparam.T;
                    break;
                }
            }
            // 计算相机0系下的标定板位姿
            board_poses[frame_id] = T_camX_cam0.inverse().eval() * T_camX;
        }
    }
}

void graph_optim(const std::vector<DetectCorner>& detected_corners,
        const std::vector<InternalParam>& internal_params,
        const std::vector<ExternalParam>& external_params,
        const std::vector<cv::Point3f>& object_points,
        const std::map<int, Eigen::Matrix4d>& board_poses_init,
        int camera_count,
        int optim_iterations,
        std::vector<ExternalParam>& optimized_external_params,
        std::vector<IterationMetadata>& iter_metadata,
        std::map<int, Eigen::Matrix4d>& final_board_poses,
        std::vector<std::pair<int, int>>& outlier_edges,
        double delta,
        int remove_iter,
        double outlier_threshold) {
    using Block = g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>>;
    // 初始化g2o优化器
    std::unique_ptr<Block::LinearSolverType> linear_solver_ptr;  // 线性求解器指针
    std::unique_ptr<Block> block_solver_ptr;  // 块求解器指针
    std::unique_ptr<g2o::OptimizationAlgorithm> opt_solver_ptr;  // 优化器solver指针
    std::unique_ptr<g2o::SparseOptimizer> optimizer_ptr;  // 优化器指针

    linear_solver_ptr = std::make_unique<g2o::LinearSolverEigen<Block::PoseMatrixType>>();
    block_solver_ptr = std::make_unique<Block>(std::move(linear_solver_ptr));
    opt_solver_ptr = std::make_unique<g2o::OptimizationAlgorithmLevenberg>(std::move(block_solver_ptr));
    optimizer_ptr = std::make_unique<g2o::SparseOptimizer>();
    optimizer_ptr->setAlgorithm(opt_solver_ptr.release());
    // optimizer_ptr->setVerbose(true);
    optimizer_ptr->setVerbose(false);

    std::map<int, std::map<int, std::vector<cv::Point2f>>> all_corners;
    for (const auto& dc : detected_corners) {
        all_corners[dc.id][dc.camera_id] = dc.corners;
    }

    std::cout << "Frame count: " << static_cast<int>(all_corners.size()) << std::endl;

    // 编号记数
    int vertex_id_count = 0, edge_id_count = 0;

    // 把内参转化为cv::Mat格式以便边使用
    std::map<int, cv::Mat> intrinsic_mat_map, dist_coeffs_map;
    for (const auto& iparam : internal_params) {
        intrinsic_mat_map[iparam.id] = cv::Mat(3, 3, CV_64F);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                intrinsic_mat_map[iparam.id].at<double>(i, j) = iparam.intrinsic(i, j);
            }
        }
        dist_coeffs_map[iparam.id] = cv::Mat(5, 1, CV_64F);
        for (int i = 0; i < 5; ++i) {
            dist_coeffs_map[iparam.id].at<double>(i, 0) = iparam.dist_coeffs(i);
        }
    }

    // 初始化相机顶点
    std::map<int, g2o::VertexSE3*> camera_vertices;
    int edge_dim = static_cast<int>(object_points.size()) * 2;
    for (int cam_id = 1; cam_id < camera_count; ++cam_id) {
        auto camera_vertex = new g2o::VertexSE3();
        camera_vertex->setId(vertex_id_count++);
        camera_vertex->setFixed(false);  // 标定板位姿可变
        // camera_vertex->setEstimate(T_iso);
        optimizer_ptr->addVertex(camera_vertex);
        // 获取初始外参
        Eigen::Matrix4d T_cam;
        for (const auto& eparam : external_params) {
            if (eparam.father_id == cam_id) {
                T_cam = eparam.T;
                break;
            }
        }
        Eigen::Isometry3d T_iso(T_cam);
        camera_vertex->setEstimate(T_iso);
        camera_vertices[cam_id] = camera_vertex;
    }

    // 逐帧添加标定板顶点和边
    int throw_count = 0;
    // 这里没法用智能指针，因为g2o内部会自己释放内存
    std::vector<MainObserveEdge*> main_edges;
    std::map<int, std::map<int, SubObserveEdge*>> sub_edges;  // frame_id -> (cam_id -> edge)
    std::map<int, g2o::VertexSE3*> board_vertices;
    for (auto& frame_pair : all_corners) {
        int frame_id = frame_pair.first;
        // 这里要滤除只有一个相机观测到的帧
        if (static_cast<int>(frame_pair.second.size()) < 2) {
            std::cout << "Frame " << frame_id << " has less than 2 camera observations, skipped." << std::endl;
            ++throw_count;
            continue;
        }

        // 查找标定板位姿初值
        Eigen::Matrix4d T_board;
        T_board = board_poses_init.at(frame_id);
        Eigen::Isometry3d T_iso(T_board);

        // 标定板顶点
        auto board_vertex = new g2o::VertexSE3();
        board_vertex->setId(vertex_id_count++);
        board_vertex->setFixed(false);  // 标定板位姿可变
        board_vertex->setEstimate(T_iso);
        optimizer_ptr->addVertex(board_vertex);
        board_vertices[frame_id] = board_vertex;
        for (auto& cam_pair : frame_pair.second) {
            int cam_id = cam_pair.first;
            const std::vector<cv::Point2f>& img_points = cam_pair.second;
            // 构造观测值
            Eigen::Vector<double, Eigen::Dynamic> measurement;
            for (auto& pt : img_points) {
                measurement.conservativeResize(measurement.size() + 2);
                measurement(measurement.size() - 2) = pt.x;
                measurement(measurement.size() - 1) = pt.y;
            }

            if (0 == cam_id) {
                // 重投影边
                auto edge = new MainObserveEdge(edge_dim,
                        intrinsic_mat_map[cam_id],
                        dist_coeffs_map[cam_id],
                        measurement);
                edge->setId(edge_id_count++);
                edge->setVertex(0, board_vertex);
                edge->setMeasurement(measurement);
                edge->setInformation(Eigen::MatrixXd::Identity(
                    measurement.size(), measurement.size()));
                if (delta > 0) {
                    auto kernel = new g2o::RobustKernelHuber();
                    kernel->setDelta(delta);
                    edge->setRobustKernel(kernel);
                }
                optimizer_ptr->addEdge(edge);
                main_edges.push_back(edge);
            } else {
                // 重投影边
                auto edge = new SubObserveEdge(edge_dim,
                        intrinsic_mat_map[cam_id],
                        dist_coeffs_map[cam_id],
                        measurement);
                edge->setId(edge_id_count++);
                edge->setVertex(0, board_vertex);
                edge->setVertex(1, camera_vertices.at(cam_id));
                edge->setMeasurement(measurement);
                edge->setInformation(Eigen::MatrixXd::Identity(
                    measurement.size(), measurement.size()));
                if (delta > 0) {
                    auto kernel = new g2o::RobustKernelHuber();
                    kernel->setDelta(delta);
                    edge->setRobustKernel(kernel);
                }
                optimizer_ptr->addEdge(edge);
                sub_edges[frame_id][cam_id] = edge;
            }
        }
    }
    std::cout << "Thrown away " << throw_count << " frames with insufficient observations." << std::endl;
    std::cout << "Edge id count: " << edge_id_count << std::endl;

    // 执行优化
    optimizer_ptr->initializeOptimization();
    // 打印图的基本信息
    std::cout << "Number of vertices: " << optimizer_ptr->vertices().size() << std::endl;
    std::cout << "Number of edges: " << optimizer_ptr->edges().size() << std::endl;
    std::cout << "Number of active vertices: " << optimizer_ptr->activeVertices().size() << std::endl;
    std::cout << "Number of active edges: " << optimizer_ptr->activeEdges().size() << std::endl;
    // optimizer_ptr->optimize(optim_iterations);

    optimizer_ptr->computeActiveErrors();
    std::cout << "Original chi2: " << optimizer_ptr->chi2()
        << "\t Average error: " << std::sqrt(optimizer_ptr->chi2() / edge_dim / edge_id_count) << std::endl;

    // 单步优化以获取每次迭代的信息
    for (int i = 0; i < optim_iterations; ++i) {
        optimizer_ptr->optimize(1);
        double chi2 = optimizer_ptr->chi2();
        // 获取优化信息
        optimizer_ptr->computeActiveErrors();
        // first dynamic_cast to a const pointer to avoid dropping const qualifiers,
        // then remove const with const_cast if a non-const pointer is required.
        const auto* lm_const = dynamic_cast<const g2o::OptimizationAlgorithmLevenberg*>(
            optimizer_ptr->algorithm());
        auto* lm = const_cast<g2o::OptimizationAlgorithmLevenberg*>(lm_const);

        if (!lm) {
            throw std::runtime_error("Optimizer is not Levenberg-Marquardt");
        }

        double lambda = lm->currentLambda();
        int lm_iter = lm->levenbergIteration();
        double average_error = std::sqrt(chi2 / edge_dim / edge_id_count);
        IterationMetadata meta;
        meta.iteration = i;
        meta.chi2 = chi2;
        meta.lambda = lambda;
        meta.levenbergIter = lm_iter;
        meta.average_error = average_error;
        iter_metadata.push_back(meta);
        std::cout << "Iteration " << i << ": chi2 = " << chi2
                  << ", lambda = " << lambda
                  << ", LM iteration = " << lm_iter
                  << ", average error = " << average_error << std::endl;
        
        if (!std::isfinite(optimizer_ptr->chi2())) {
            std::cerr << "Chi2 exploded, stop optimization" << std::endl;
            break;
        }
        if (i == remove_iter) {
            int dim = object_points.size() * 2;
            for (auto& edge : sub_edges) {
                int frame_id = edge.first;
                for (auto& cam_edge_pair : edge.second) {
                    int cam_id = cam_edge_pair.first;
                    auto* sub_edge = cam_edge_pair.second;
                    double average_err = std::sqrt(sub_edge->chi2() / dim);
                    if (average_err > outlier_threshold) {
                        sub_edge->setLevel(1);  // 标记为不参与后续优化
                        outlier_edges.push_back(std::make_pair(frame_id, cam_id));
                        std::cout << "Removing outlier edge at frame " << frame_id
                                  << ", camera " << cam_id
                                  << " with average error = " << average_err << std::endl;
                    }
                    if (std::isnan(average_err) || std::isinf(average_err)) {
                        auto info = sub_edge->information();
                        if (!info.allFinite()) {
                            std::cout << "Edge information matrix has non-finite values." << std::endl;
                        }
                        
                    }
                }
            }
            std::cout << "Active edges after outlier removal: " << optimizer_ptr->activeEdges().size() << std::endl;
        }
    }
    // 提取优化后的外参
    optimizer_ptr->computeActiveErrors();
    for (int cam_id = 1; cam_id < camera_count; ++cam_id) {
        ExternalParam eparam;
        eparam.father_id = cam_id;
        eparam.child_id = 0;
        Eigen::Isometry3d T_iso = camera_vertices.at(cam_id)->estimate();
        eparam.T = T_iso.matrix();
        optimized_external_params.push_back(eparam);
    }
    // 提取每一帧标定板位姿
    for (auto& board_pair : board_vertices) {
        int frame_id = board_pair.first;
        Eigen::Isometry3d T_iso = board_pair.second->estimate();
        final_board_poses[frame_id] = T_iso.matrix();
    }
}

void dump_calib_result(const std::string& filename,
        const std::vector<InternalParam>& internal_params,
        const std::vector<ExternalParam>& external_params,
        const std::vector<ExternalParam>& rough_external_params) {
    YAML::Emitter out;

    out << YAML::BeginMap;

    out << YAML::Key << "internal_params" << YAML::Value << YAML::BeginSeq;
    for (const auto& iparam : internal_params) {
        out << YAML::BeginMap;
        out << YAML::Key << "id" << YAML::Value << iparam.id;
        out << YAML::Key << "intrinsic" << YAML::Value << YAML::BeginSeq;
        for (int i = 0; i < 3; ++i) {
            out << YAML::Flow << YAML::BeginSeq;
            for (int j = 0; j < 3; ++j) {
                out << iparam.intrinsic(i, j);
            }
            out << YAML::EndSeq;
        }
        out << YAML::EndSeq;
        out << YAML::Key << "dist_coeffs" << YAML::Value << YAML::BeginSeq;
        out << YAML::Flow << YAML::BeginSeq;
        for (int i = 0; i < 5; ++i) {
            out << iparam.dist_coeffs(i);
        }
        out << YAML::EndSeq;
        out << YAML::EndSeq;
        out << YAML::EndMap;
    }
    out << YAML::EndSeq;

    out << YAML::Key << "external_params" << YAML::Value << YAML::BeginSeq;
    for (const auto& eparam : external_params) {
        out << YAML::BeginMap;
        out << YAML::Key << "father_id" << YAML::Value << eparam.father_id;
        out << YAML::Key << "child_id" << YAML::Value << eparam.child_id;
        out << YAML::Key << "T_father_child" << YAML::Value << YAML::BeginSeq;
        for (int i = 0; i < 4; ++i) {
            out << YAML::Flow << YAML::BeginSeq;
            for (int j = 0; j < 4; ++j) {
                out << eparam.T(i, j);
            }
            out << YAML::EndSeq;
        }
        out << YAML::EndSeq;
        out << YAML::Key << "T_father_child_rough" << YAML::Value << YAML::BeginSeq;
        for (int i = 0; i < 4; ++i) {
            out << YAML::Flow << YAML::BeginSeq;
            for (int j = 0; j < 4; ++j) {
                out << rough_external_params
                    .at(eparam.father_id - 1).T(i, j);
            }
            out << YAML::EndSeq;
        }
        out << YAML::EndSeq;
        out << YAML::EndMap;
    }
    out << YAML::EndSeq;

    out << YAML::EndMap;

    // 写入文件
    std::ofstream fout(filename);
    fout << out.c_str();
    fout.close();
}

void dump_iteration_metadata_csv(
        const std::string& filename,
        const std::vector<IterationMetadata>& iter_metadata) {
    std::ofstream fout(filename);
    fout << "iteration,chi2,lambda,levenbergIter,average_error\n";
    for (const auto& meta : iter_metadata) {
        fout << meta.iteration << ","
             << meta.chi2 << ","
             << meta.lambda << ","
             << meta.levenbergIter << ","
             << meta.average_error << "\n";
    }
    fout.close();
}

int main(int argc, char** argv) {
    // 从命令行获取config.yaml的路径
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config.yaml>" << std::endl;
        return 1;
    }

    const std::string root_path = argv[1];
    const std::string config_path = root_path + "/config.yaml";
    std::cout << "使用配置文件: " << config_path << std::endl;
    YAML::Node config;
    try {
        config = YAML::LoadFile(config_path);
        (void)config;  // 暂存配置，后续可扩展使用
    } catch (const std::exception& e) {
        std::cerr << "读取配置文件失败: " << e.what() << std::endl;
        return 1;
    }

    std::string corner_out_path = root_path + '/' + config["corner_out_path"].as<std::string>();
    int camera_count = config["camera_count"].as<int>();


    std::vector<DetectCorner> detected_corners;
    load_detected_corners(corner_out_path, detected_corners);

    // 获取标定板参数
    std::vector<int> corner_size_vec = config["corner_size"].as<std::vector<int>>();
    cv::Size corner_size(corner_size_vec[0], corner_size_vec[1]);
    double corner_edge_length = config["corner_edge_length"].as<double>();
    int height = config["camera_params"][0]["height"].as<int>();
    int width = config["camera_params"][0]["width"].as<int>();
    int rotation_flag = config["image_rotation"].as<int>();
    cv::Size camera_size;
    if (rotation_flag % 2 != 0) {
        camera_size = cv::Size(width, height);
    } else {
        camera_size = cv::Size(height, width);
    }
    double delta = config["delta"].as<double>();
    int remove_iter = config["remove_iter"].as<int>();
    double outlier_threshold = config["outlier_threshold"].as<double>();

    // 标定板系下格点坐标
    std::vector<cv::Point3f> object_points;
    for (int y = 0; y < corner_size.height; ++y) {
        for (int x = 0; x < corner_size.width; ++x) {
            object_points.emplace_back(x * corner_edge_length, y * corner_edge_length, 0.0);
        }
    }

    // 粗标定
    std::vector<InternalParam> internal_params;
    std::vector<ExternalParam> external_params;
    std::map<int, Eigen::Matrix4d> board_poses;
    rough_calib(detected_corners, internal_params, external_params, board_poses,
            object_points, camera_size, camera_count);

    // 要记得给边的静态成员变量corners赋值
    MainObserveEdge::corners = object_points;
    SubObserveEdge::corners = object_points;

    // 图优化
    std::vector<ExternalParam> optimized_external_params;
    std::vector<IterationMetadata> iter_metadata;
    std::map<int, Eigen::Matrix4d> optmi_board_poses;
    std::vector<std::pair<int, int>> outlier_edges;
    int optim_iterations = config["optim_iterations"].as<int>();
    graph_optim(detected_corners, internal_params, external_params,
            object_points, board_poses, camera_count, optim_iterations,
            optimized_external_params, iter_metadata, optmi_board_poses, outlier_edges,
            delta, remove_iter, outlier_threshold);
    // 可视化优化结果
    visualize_graph_optimization_corners(detected_corners,
        internal_params, optimized_external_params, external_params,
        object_points, optmi_board_poses, board_poses, outlier_edges,
        camera_size, corner_size, camera_count);
    // 保存标定结果
    std::string calib_out_path = root_path + '/' + config["calib_result_path"].as<std::string>();
    dump_calib_result(calib_out_path, internal_params, optimized_external_params, external_params);
    std::cout << "标定结果已保存到: " << calib_out_path << std::endl;
    // 保存迭代元数据
    std::string meta_data_path = root_path + '/' + config["meta_data_path"].as<std::string>();
    dump_iteration_metadata_csv(meta_data_path, iter_metadata);
    std::cout << "迭代元数据已保存到: " << meta_data_path << std::endl;

    return 0;
}