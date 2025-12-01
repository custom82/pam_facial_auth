cmake_minimum_required(VERSION 3.16)
project(pam_facial_auth LANGUAGES CXX)

option(ENABLE_CUDA   "Enable CUDA DNN backend"   OFF)
option(ENABLE_OPENCL "Enable OpenCL DNN backend" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED core imgproc imgcodecs videoio objdetect dnn face)

find_library(PAM_LIB pam REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(ENABLE_CUDA)
    add_definitions(-DENABLE_CUDA)
endif()

if(ENABLE_OPENCL)
    add_definitions(-DENABLE_OPENCL)
endif()

add_library(facialauth SHARED
    src/libfacialauth.cpp
)

target_link_libraries(facialauth
    ${OpenCV_LIBS}
)

add_library(pam_facial_auth SHARED
    src/pam_facial_auth.cpp
)

target_link_libraries(pam_facial_auth
    facialauth
    ${PAM_LIB}
    ${OpenCV_LIBS}
)

set_target_properties(pam_facial_auth PROPERTIES
    OUTPUT_NAME "pam_facial_auth"
    PREFIX ""           # per libpam_xxx.so
)

add_executable(facial_capture
    src/facial_capture.cpp
)

target_link_libraries(facial_capture
    facialauth
    ${OpenCV_LIBS}
)

add_executable(facial_training
    src/facial_training.cpp
)

target_link_libraries(facial_training
    facialauth
    ${OpenCV_LIBS}
)

add_executable(facial_test
    src/facial_test.cpp
)

target_link_libraries(facial_test
    facialauth
    ${OpenCV_LIBS}
)
