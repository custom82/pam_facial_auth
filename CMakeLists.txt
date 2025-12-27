cmake_minimum_required(VERSION 3.10)
project(pam_facial_auth LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Forza l'output dei file binari nella cartella di build corrente
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Cerca le dipendenze
find_package(OpenCV REQUIRED COMPONENTS core imgproc objdetect videoio highgui face)
find_package(PkgConfig REQUIRED)
pkg_check_modules(PAM REQUIRED pam)

add_definitions(-DHAVE_OPENCV=1)

# 1. Libreria Core
add_library(facialauth SHARED src/libfacialauth.cpp)
target_include_directories(facialauth PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(facialauth PRIVATE ${OpenCV_LIBRARIES})

# 2. Modulo PAM (Plugin)
add_library(pam_facial_auth MODULE src/pam_facial_auth.cpp)
target_link_libraries(pam_facial_auth PRIVATE facialauth ${PAM_LIBRARIES})
set_target_properties(pam_facial_auth PROPERTIES PREFIX "" LIBRARY_OUTPUT_NAME "pam_facial_auth")

# 3. Tool CLI
add_executable(facial_capture src/facial_capture.cpp)
target_link_libraries(facial_capture PRIVATE facialauth)

add_executable(facial_training src/facial_training.cpp)
target_link_libraries(facial_training PRIVATE facialauth)

add_executable(facial_test src/facial_test.cpp)
target_link_libraries(facial_test PRIVATE facialauth)
