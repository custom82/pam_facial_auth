cmake_minimum_required(VERSION 3.16)
project(pam_facial_auth VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# Cerca OpenCV con tutti i moduli necessari
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio face dnn highgui)
find_package(PAM REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})

# 1. Libreria Core
add_library(facialauth SHARED src/libfacialauth.cpp)
target_link_libraries(facialauth PRIVATE ${OpenCV_LIBS})

# 2. Modulo PAM
add_library(pam_facial_auth SHARED src/pam_facial_auth.cpp)
set_target_properties(pam_facial_auth PROPERTIES PREFIX "")
target_link_libraries(pam_facial_auth PRIVATE facialauth PAM::PAM ${OpenCV_LIBS})

# 3. Tool CLI (Cattura, Training, Test)
add_executable(facial_capture src/facial_capture.cpp)
target_link_libraries(facial_capture PRIVATE facialauth ${OpenCV_LIBS})

add_executable(facial_training src/facial_training.cpp)
target_link_libraries(facial_training PRIVATE facialauth ${OpenCV_LIBS})

add_executable(facial_test src/facial_test.cpp)
target_link_libraries(facial_test PRIVATE facialauth ${OpenCV_LIBS})

# Installazione
install(TARGETS pam_facial_auth DESTINATION /usr/lib64/security)
install(TARGETS facial_capture facial_training facial_test DESTINATION /usr/bin)
install(FILES pam_facial.conf DESTINATION /etc/security)
