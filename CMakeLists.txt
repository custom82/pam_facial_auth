cmake_minimum_required(VERSION 3.10)
project(pam_facial_auth CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================
# FIND PACKAGES
# ============================================================

find_package(OpenCV REQUIRED)
find_package(PAM REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${PAM_INCLUDE_DIRS}
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

# PAM modules MUST NOT have the "lib" prefix
set_target_properties(pam_facial_auth PROPERTIES
    PREFIX ""
    OUTPUT_NAME "pam_facial_auth"
)

target_link_libraries(pam_facial_auth
    facialauth
    ${OpenCV_LIBS}
    ${PAM_LIBRARIES}
)

# ============================================================
# EXECUTABLES (CLI)
# ============================================================

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

# ============================================================
# NO INSTALL SECTION
# (Installazioni gestite da ebuild / spec / deb rules)
# ============================================================

# intentionally empty

# ============================================================
#  NO INSTALL RULES — installation handled by packaging tools
# ============================================================

message(STATUS "No installation from CMake.")
message(STATUS "Installation must be done using:")
message(STATUS "  * Gentoo:     ebuild")
message(STATUS "  * RPM:        .spec file")
message(STATUS "  * Debian:     debian/rules")
