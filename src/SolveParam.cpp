#include <string>
#include <vector>
#include <iostream>
#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include <g2o/types/slam3d/vertex_se3.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
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

void rough_calib(const std::vector<DetectCorner>& detected_corners,
        std::vector<InternalParam>& internal_params,
        std::vector<ExternalParam>& external_params,
        const std::vector<cv::Point3f>& object_points,
        int camera_count) {
    std::vector<std::map<int, std::vector<cv::Point2f>>> all_corners(camera_count);
    for (const auto& dc : detected_corners) {
        all_corners[dc.camera_id][dc.id] = dc.corners;
    }
    // 对每个相机进行内参标定
    std::map<int, cv::Mat> camera_matrices;
    std::map<int, cv::Mat> dist_coeffs_map;
    for (int cam_id = 0; cam_id < camera_count; ++cam_id) {
        std::vector<std::vector<cv::Point3f>> obj_pts_vec;
        std::vector<std::vector<cv::Point2f>> img_pts_vec;
        for (const auto& pair : all_corners[cam_id]) {
            obj_pts_vec.push_back(std::vector<cv::Point3f>(object_points.begin(), object_points.end()));
            img_pts_vec.push_back(pair.second);
        }

        cv::Mat camera_matrix = cv::Mat::eye(3, 3, CV_64F);
        cv::Mat dist_coeffs = cv::Mat::zeros(5, 1, CV_64F);
        std::vector<cv::Mat> rvecs, tvecs;
        cv::TermCriteria term_crit(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 100, DBL_EPSILON);
        double rms = cv::calibrateCamera(obj_pts_vec, img_pts_vec, cv::Size(800, 600),
                camera_matrix, dist_coeffs, rvecs, tvecs,
                0, term_crit);
        std::cout << "Camera " << cam_id << " calibrated with RMS error = " << rms << std::endl;

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
    cv::Mat cam1_matrix = camera_matrices[0];
    cv::Mat dist1_coeffs = dist_coeffs_map[0];
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
                img_pts_vec1.push_back(pair.second);
                img_pts_vec2.push_back(corners2);
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
            cv::Size(800, 600),
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
        external_params.push_back(eparam);
    }
}

void graph_optim(const std::vector<DetectCorner>& detected_corners,
        std::vector<InternalParam>& internal_params,
        std::vector<ExternalParam>& external_params,
        const std::vector<cv::Point3f>& object_points,
        int camera_count,
        int optim_iterations,
        std::vector<ExternalParam>& optimized_external_params,
        std::vector<IterationMetadata>& iter_metadata) {
    using Block = g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>>;
    // 初始化g2o优化器
    // Block::LinearSolverType* linear_solver =
    //     new g2o::LinearSolverEigen<Block::PoseMatrixType>();

    // auto* block_solver = new Block(linear_solver);

    // auto* algorithm =
    //     new g2o::OptimizationAlgorithmLevenberg(block_solver);

    // auto* optimizer = new g2o::SparseOptimizer();
    // optimizer->setAlgorithm(algorithm);
    // optimizer->setVerbose(false);
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

    // 设置一个标定板顶点的初始值
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 1>(0, 3) = Eigen::Vector3d(0.0, 0.0, 0.1);  // 初始位置放在z轴正方向0.1米处
    Eigen::Isometry3d T_iso(T);

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
    // // 为camera 0设置固定顶点
    // auto camera0_vertex = std::make_shared<g2o::VertexSE3>();
    // camera0_vertex->setId(vertex_id_count++);
    // camera0_vertex->setFixed(true);  // 相机0位姿固定
    // Eigen::Isometry3d T0_iso = Eigen::Isometry3d::Identity();
    // camera0_vertex->setEstimate(T0_iso);
    // optimizer_ptr->addVertex(camera0_vertex.get());
    // camera_vertices[0] = camera0_vertex;

    // 逐帧添加标定板顶点和边
    int throw_count = 0;
    // 这里没法用智能指针，因为g2o内部会自己释放内存
    std::vector<MainObserveEdge*> main_edges;
    std::vector<SubObserveEdge*> sub_edges;
    std::vector<g2o::VertexSE3*> board_vertices;
    for (auto& frame_pair : all_corners) {
        int frame_id = frame_pair.first;
        // 这里要滤除只有一个相机观测到的帧
        if (static_cast<int>(frame_pair.second.size()) < 2) {
            std::cout << "Frame " << frame_id << " has less than 2 camera observations, skipped." << std::endl;
            ++throw_count;
            continue;
        }
        // std::vector<std::shared_ptr<g2o::VertexSE3>> board_vertices, cam_vertices;
        // 标定板顶点
        auto board_vertex = new g2o::VertexSE3();
        board_vertex->setId(vertex_id_count++);
        board_vertex->setFixed(false);  // 标定板位姿可变
        board_vertex->setEstimate(T_iso);
        optimizer_ptr->addVertex(board_vertex);
        board_vertices.push_back(board_vertex);
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

            // // 重投影边
            // auto edge = std::make_shared<SubObserveEdge>(edge_dim,
            //         intrinsic_mat_map[cam_id],
            //         dist_coeffs_map[cam_id],
            //         measurement);
            // edge->setId(edge_id_count++);
            // edge->setVertex(0, board_vertex.get());
            // edge->setVertex(1, camera_vertices.at(cam_id).get());
            // edge->setMeasurement(measurement);
            // edge->setInformation(Eigen::MatrixXd::Identity(
            //     measurement.size(), measurement.size()));
            // optimizer_ptr->addEdge(edge.get());
            // edges.push_back(edge);

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
                optimizer_ptr->addEdge(edge);
                sub_edges.push_back(edge);
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

    // 单步优化以获取每次迭代的信息
    for (int i = 0; i < optim_iterations; ++i) {
        optimizer_ptr->optimize(1);
        double chi2 = optimizer_ptr->chi2();
        // 获取优化信息
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
    }
    // 提取优化后的外参
    for (int cam_id = 1; cam_id < camera_count; ++cam_id) {
        ExternalParam eparam;
        eparam.father_id = cam_id;
        eparam.child_id = 0;
        Eigen::Isometry3d T_iso = camera_vertices.at(cam_id)->estimate();
        eparam.T = T_iso.matrix();
        optimized_external_params.push_back(eparam);
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
    std::vector<int> corner_size = config["corner_size"].as<std::vector<int>>();
    double corner_edge_length = config["corner_edge_length"].as<double>();

    // 标定板系下格点坐标
    std::vector<cv::Point3f> object_points;
    for (int y = 0; y < corner_size[1]; ++y) {
        for (int x = 0; x < corner_size[0]; ++x) {
            object_points.emplace_back(x * corner_edge_length, y * corner_edge_length, 0.0);
        }
    }

    // 粗标定
    std::vector<InternalParam> internal_params;
    std::vector<ExternalParam> external_params;
    rough_calib(detected_corners, internal_params, external_params,
            object_points, camera_count);

    // 要记得给边的静态成员变量corners赋值
    MainObserveEdge::corners = object_points;
    SubObserveEdge::corners = object_points;

    // 图优化
    std::vector<ExternalParam> optimized_external_params;
    std::vector<IterationMetadata> iter_metadata;
    int optim_iterations = config["optim_iterations"].as<int>();
    graph_optim(detected_corners, internal_params, external_params,
            object_points, camera_count, optim_iterations,
            optimized_external_params, iter_metadata);
    // 保存标定结果
    std::string calib_out_path = root_path + '/' + config["calib_result_path"].as<std::string>();
    dump_calib_result(calib_out_path, internal_params, optimized_external_params, external_params);
    std::cout << "标定结果已保存到: " << calib_out_path << std::endl;
    return 0;
}