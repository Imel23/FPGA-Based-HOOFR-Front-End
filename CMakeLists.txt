cmake_minimum_required(VERSION 3.0)
PROJECT(HOOFR_EXTRACTOR_FPGA LANGUAGES CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -pthread -fPIC -fopenmp")

FIND_PACKAGE(OpenCV 4.0.0 REQUIRED)
MESSAGE("OpenCV version: ${OpenCV_VERSION}")
find_package(Eigen3 3.1.0 REQUIRED)

include_directories(
${OpenCV_INCLUDE_DIRS}
${EIGEN3_INCLUDE_DIR}
./include/
)
    
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)

add_executable(host host.cpp
src/AOCL_Utils.cpp
)

target_link_libraries(host 
${OpenCV_LIBS}
-lOpenCL
-pthread
)
