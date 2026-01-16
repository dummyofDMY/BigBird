cmake_minimum_required(VERSION 3.14)
project(BigBird LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Threads REQUIRED)
find_package(g2o REQUIRED)
find_package(Eigen3 REQUIRED)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "Eigen version: ${EIGEN3_VERSION}")
message(STATUS "yaml-cpp version: ${YAML_CPP_VERSION}")
message(STATUS "g2o version: ${G2O_VERSION}")

add_executable(detect_corner
	src/DetectCorner.cpp
	src/Camera.cpp
)

add_executable(solve_param
	src/SolveParam.cpp
	src/Graph.cpp
)

target_include_directories(detect_corner PRIVATE include)
target_include_directories(solve_param PRIVATE include
  ${EIGEN3_INCLUDE_DIR}
  ${g2o_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(detect_corner PRIVATE ${OpenCV_LIBS} yaml-cpp Threads::Threads)
target_link_libraries(solve_param PRIVATE
  Eigen3::Eigen
  g2o::core
  g2o::types_slam3d
  g2o::solver_eigen
  g2o::stuff
  yaml-cpp
  ${OpenCV_LIBS}
)
