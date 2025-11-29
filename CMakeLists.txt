cmake_minimum_required(VERSION 3.10)
project(pam_facial_auth CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================
# FIND PACKAGES
# ============================================================

find_package(OpenCV REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)

# ============================================================
# LIBRARY: libfacialauth.so
# ============================================================

add_library(facialauth SHARED
    src/libfacialauth.cpp
)

target_link_libraries(facialauth
    ${OpenCV_LIBS}
)

# ============================================================
# PAM MODULE: pam_facial_auth.so
# ============================================================

add_library(pam_facial_auth MODULE
    src/pam_facial_auth.cpp
)

set_target_properties(pam_facial_auth PROPERTIES
    PREFIX ""
    OUTPUT_NAME "pam_facial_auth"
)

target_link_libraries(pam_facial_auth
    facialauth
    ${OpenCV_LIBS}
    pam
)

# ============================================================
# EXECUTABLES (CLI tools)
# ============================================================

add_executable(facial_capture src/facial_capture.cpp)
target_link_libraries(facial_capture facialauth ${OpenCV_LIBS})

add_executable(facial_training src/facial_training.cpp)
target_link_libraries(facial_training facialauth ${OpenCV_LIBS})

add_executable(facial_test src/facial_test.cpp)
target_link_libraries(facial_test facialauth ${OpenCV_LIBS})

# ============================================================
# NO INSTALL (handled by ebuild / .spec / deb rules)
# ============================================================
