#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
echo "脚本目录: ${SCRIPT_DIR}"
PROJECT_DIR=$(dirname ${SCRIPT_DIR})
echo "项目目录: ${PROJECT_DIR}"

${PROJECT_DIR}/build/solve_param ${PROJECT_DIR}