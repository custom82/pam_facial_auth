cmake_minimum_required(VERSION 3.16)
project(pam_facial_auth VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# Cerca OpenCV
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio face dnn highgui)

# --- CORREZIONE PER PAM SU GENTOO ---
find_library(PAM_LIBRARY NAMES pam REQUIRED)
find_path(PAM_INCLUDE_DIR NAMES security/pam_appl.h REQUIRED)
# ------------------------------------

include_directories(${CMAKE_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS} ${PAM_INCLUDE_DIR})

# 1. Libreria Core
add_library(facialauth SHARED src/libfacialauth.cpp)
target_link_libraries(facialauth PRIVATE ${OpenCV_LIBS})

# 2. Modulo PAM (usiamo PAM_LIBRARY trovata sopra)
add_library(pam_facial_auth SHARED src/pam_facial_auth.cpp)
set_target_properties(pam_facial_auth PROPERTIES PREFIX "")
target_link_libraries(pam_facial_auth PRIVATE facialauth ${PAM_LIBRARY} ${OpenCV_LIBS})

# 3. Tool CLI
add_executable(facial_capture src/facial_capture.cpp)
target_link_libraries(facial_capture PRIVATE facialauth ${OpenCV_LIBS})

add_executable(facial_training src/facial_training.cpp)
target_link_libraries(facial_training PRIVATE facialauth ${OpenCV_LIBS})

add_executable(facial_test src/facial_test.cpp)
target_link_libraries(facial_test PRIVATE facialauth ${OpenCV_LIBS})


