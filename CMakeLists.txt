cmake_minimum_required(VERSION 3.20)
project(pam_facial_auth)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED COMPONENTS core objdetect highgui videoio imgproc face dnn)
find_package(PAM REQUIRED)

add_library(facialauth SHARED src/libfacialauth.cpp)
target_link_libraries(facialauth PRIVATE ${OpenCV_LIBS})

add_library(pam_facial_auth MODULE src/pam_facial_auth.cpp)
target_link_libraries(pam_facial_auth PRIVATE facialauth ${PAM_LIBRARIES})
set_target_properties(pam_facial_auth PROPERTIES PREFIX "")

add_executable(facial_training src/facial_training.cpp)
target_link_libraries(facial_training PRIVATE facialauth)

add_executable(facial_test src/facial_test.cpp)
target_link_libraries(facial_test PRIVATE facialauth)
