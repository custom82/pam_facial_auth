cmake_minimum_required(VERSION 3.26)
project(pam_facial_auth LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED core imgproc imgcodecs videoio objdetect dnn face)

find_library(PAM_LIBRARY pam REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

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
    ${PAM_LIBRARY}
)

add_executable(facial_capture
    src/facial_capture.cpp
)
target_link_libraries(facial_capture facialauth)

add_executable(facial_training
    src/facial_training.cpp
)
target_link_libraries(facial_training facialauth)

add_executable(facial_test
    src/facial_test.cpp
)
target_link_libraries(facial_test facialauth)
