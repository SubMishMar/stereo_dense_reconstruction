cmake_minimum_required(VERSION 2.8.3)
project(stereo_dense_reconstruction)

find_package(catkin REQUIRED COMPONENTS
        pcl_conversions
        pcl_ros
        roscpp
        sensor_msgs
        velodyne_pointcloud
        message_filters
        cv_bridge
        tf
        tf_conversions
        message_generation
        image_transport
        )

find_package(OpenCV REQUIRED)
set(LIBELAS_SRC_DIR src/elas)
set(CMAKE_CXX_FLAGS "-msse3")
set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 
find_package(Boost 1.54.0 COMPONENTS system filesystem program_options REQUIRED)

catkin_package(
        INCLUDE_DIRS
        CATKIN_DEPENDS
        DEPENDS
)



FILE(GLOB LIBELAS_SRC_FILES "src/elas/*.cpp")

include_directories(
        ${catkin_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${LIBELAS_SRC_DIR}
        ${Boost_INCLUDE_DIRS}
        ${INC_DIR}
)

add_executable(dense_reconstruction
        src/dense_reconstruction.cpp
        ${LIBELAS_SRC_FILES}
)
target_link_libraries(dense_reconstruction ${OpenCV_LIBS} ${catkin_LIBRARIES})
