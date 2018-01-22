cmake_minimum_required (VERSION 2.8)
cmake_policy(SET CMP0015 OLD)

project (track_targets)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("libs/cscore/include")
include_directories("libs/ntcore/include")
include_directories("libs/wpiutil/include")

link_directories("/home/ubuntu/projects/github/teamblitz/Jetson_TK1_2017/libs/cscore/Linux/arm")
link_directories("/home/ubuntu/projects/github/teamblitz/Jetson_TK1_2017/libs/ntcore/Linux/arm")
link_directories("/home/ubuntu/projects/github/teamblitz/Jetson_TK1_2017/libs/wpiutil/Linux/arm")

add_definitions(-std=c++11)

aux_source_directory(src CV_SOURCES)

add_executable(cv_track_targets ${CV_SOURCES})

target_link_libraries(cv_track_targets ${OpenCV_LIBS} cscore ntcore wpiutil)
    

