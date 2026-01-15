#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>

struct DetectCorner
{
    int id;
    int camera_id;
    std::vector<cv::Point2f> corners;
};

void dump_detected_corners(const std::vector<DetectCorner>& detected_corners, const std::string& filename)
{
    YAML::Emitter out;
    out << YAML::BeginSeq;
    for (const auto& dc : detected_corners) {
        out << YAML::BeginMap;
        out << YAML::Key << "id" << YAML::Value << dc.id;
        out << YAML::Key << "camera_id" << YAML::Value << dc.camera_id;
        out << YAML::Key << "corners" << YAML::Value << YAML::BeginSeq;
        for (const auto& pt : dc.corners) {
            out << YAML::Flow << YAML::BeginSeq << pt.x << pt.y << YAML::EndSeq;
        }
        out << YAML::EndSeq; // corners
        out << YAML::EndMap;
    }
    out << YAML::EndSeq;

    std::ofstream fout(filename);
    fout << out.c_str();
    fout.close();
}

void load_detected_corners(const std::string& filename, std::vector<DetectCorner>& detected_corners)
{
    detected_corners.clear();
    YAML::Node data = YAML::LoadFile(filename);
    for (const auto& node : data) {
        DetectCorner dc;
        dc.id = node["id"].as<int>();
        dc.camera_id = node["camera_id"].as<int>();
        for (const auto& pt_node : node["corners"]) {
            float x = pt_node[0].as<float>();
            float y = pt_node[1].as<float>();
            dc.corners.emplace_back(x, y);
        }
        detected_corners.push_back(dc);
    }
}