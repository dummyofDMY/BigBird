#include <atomic>
#include <csignal>
#include <memory>
#include <vector>
#include <fstream>
#include <iostream>
#include <yaml-cpp/yaml.h>

#include "Camera.hpp"
#include "CornerData.hpp"

static std::atomic<bool> g_stop_requested{false};

static void handle_sigint(int)
{
    g_stop_requested.store(true);
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
    int rotation_flag = config["image_rotation"].as<int>();
    std::vector<int> corner_size_vec = config["corner_size"].as<std::vector<int>>();
    cv::Size corner_size(corner_size_vec[0], corner_size_vec[1]);

    std::vector<std::vector<int>> edge_count(camera_count, std::vector<int>(camera_count, 0));

    std::vector<DetectCorner> detected_corners;
    std::signal(SIGINT, handle_sigint);
    bool do_not_save = false;
    try {
        std::map<int, std::unique_ptr<Camera>> cameras;
        std::vector<int> camera_ids;
        int id_count = 0;
        int min_fps = 1e6;
        for (int i = 0; i < camera_count; ++i) {
            int cam_id = config["camera_params"][i]["id"].as<int>();
            int width = config["camera_params"][i]["width"].as<int>();
            int height = config["camera_params"][i]["height"].as<int>();
            int fps = config["camera_params"][i]["fps"].as<int>();
            int exposure = config["camera_params"][i]["exposure"].as<int>();
            std::string fourcc = config["camera_params"][i]["fourcc"].as<std::string>();
            cameras.emplace(cam_id, std::make_unique<Camera>(
                cam_id, width, height, fps, fourcc, corner_size, exposure, rotation_flag));
            camera_ids.push_back(cam_id);
            if (fps < min_fps)
                min_fps = fps;
        }
        int wait_time = 1000 / min_fps;

        std::sort(camera_ids.begin(), camera_ids.end());
        std::map<int, int> camera_id_remap;
        for (size_t i = 0; i < camera_ids.size(); ++i) {
            camera_id_remap[camera_ids[i]] = static_cast<int>(i);
        }

        int key_input = 0;
        int now_id = 0;
        cv::namedWindow("Control", cv::WINDOW_AUTOSIZE);
        while (key_input != 27 && !g_stop_requested.load())
        {
            if ('g' == key_input || 'G' == key_input) {
                std::vector<DetectCorner> now_detected_corners;
                // 计时，太过不同步将丢弃该帧
                auto start_time = std::chrono::high_resolution_clock::now();
                for (auto& cam_pair : cameras) {
                    cam_pair.second->grab();
                }
                auto mid_time = std::chrono::high_resolution_clock::now();
                int elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(mid_time - start_time).count();
                if (elapsed_ms > wait_time) {
                    std::cout << "抓取图像超时，丢弃该次抓取，耗时 " << elapsed_ms << " ms" << std::endl;
                    continue;
                }
                for (auto& cam_pair : cameras) {
                    int camera_id = cam_pair.first;
                    std::vector<cv::Point2f> corners = cam_pair.second->get_corner();
                    if (!corners.empty()) {
                        DetectCorner dc;
                        dc.id = now_id;
                        dc.camera_id = camera_id_remap[camera_id];
                        dc.corners.assign(corners.begin(), corners.end());
                        now_detected_corners.push_back(dc);
                    }
                }
                if (static_cast<int>(now_detected_corners.size()) > 0) {
                    now_id++;
                    detected_corners.insert(detected_corners.end(),
                                            now_detected_corners.begin(),
                                            now_detected_corners.end());
                    for (size_t i = 0; i < now_detected_corners.size(); ++i) {
                        for (size_t j = i + 1; j < now_detected_corners.size(); ++j) {
                            int id1 = now_detected_corners[i].camera_id;
                            int id2 = now_detected_corners[j].camera_id;
                            edge_count[id1][id2]++;
                            // edge_count[id2][id1]++;
                        }
                    }
                    for (int i = 0; i < static_cast<int>(now_detected_corners.size()); ++i) {
                        int id = now_detected_corners[i].camera_id;
                        edge_count[id][id]++; // 自己到自己的边，表示该相机检测到的次数
                    }
                    std::cout << "检测到 " << now_detected_corners.size() << " 个相机的角点" << std::endl;
                    std::cout << "总共检测到 " << now_id << " 组角点数据。" << std::endl;
                } else {
                    std::cout << "未检测到角点" << std::endl;
                }
            } else if ('q' == key_input || 'Q' == key_input) {
                do_not_save = true;
                break;
            } else {
                cv::Mat control_image(400, 500, CV_8UC3, cv::Scalar(50, 50, 50));
                cv::putText(control_image, "Press 'g' to get corners", cv::Point(10, 50),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
                cv::putText(control_image, "Press 'ESC' to exit", cv::Point(10, 80),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
                cv::putText(control_image, "Press 'q' to exit without saving", cv::Point(10, 110),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
                cv::putText(control_image, "Detected sets: " + std::to_string(now_id), cv::Point(10, 140),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
                // 显示各相机边的检测次数
                int pix_stride = 30;
                cv::Point origin = cv::Point(10, 170);
                // 绘制表头
                for (int i = 0; i < camera_count; ++i) {
                    std::string text = "C" + std::to_string(i);
                    cv::putText(control_image, text, cv::Point(origin.x + pix_stride * (i + 1), origin.y),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
                    cv::putText(control_image, text, cv::Point(origin.x, origin.y + pix_stride * (i + 1)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
                }
                for (int i = 0; i < camera_count; ++i) {
                    for (int j = i; j < camera_count; ++j) {
                        std::string text = std::to_string(edge_count[i][j]);
                        cv::putText(control_image, text, cv::Point(origin.x + pix_stride * (j + 1), origin.y + pix_stride * (i + 1)),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 1);
                    }
                }
                cv::imshow("Control", control_image);

                // 显示各相机的实时画面
                for (auto& cam_pair : cameras) {
                    cam_pair.second->visualize();
                }
            }
            key_input = cv::waitKey(11);
        }
        if (g_stop_requested.load()) {
            std::cout << "收到 SIGINT，正在保存当前检测数据..." << std::endl;
        }
        if (!detected_corners.empty() && !do_not_save) {
            dump_detected_corners(detected_corners, corner_out_path);
            std::cout << "检测数据已保存到 " << corner_out_path << std::endl;
        } else if (do_not_save) {
            std::cout << "用户选择不保存检测数据。" << std::endl;
        } else {
            std::cout << "没有检测到任何角点数据，未保存文件。" << std::endl;
        }
        // 销毁资源
        cameras.clear();
        cv::destroyAllWindows();
        
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
    return 0;
}