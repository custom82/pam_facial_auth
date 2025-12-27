cmake_minimum_required(VERSION 3.22)
project(pam_facial_auth LANGUAGES CXX)

# Force C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# --- RPATH FIX FOR GENTOO QA ---
# Skip RPATH for the install tree
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)
set(CMAKE_SKIP_INSTALL_RPATH TRUE)
set(CMAKE_SKIP_RPATH TRUE)

# Optional: ensure that even during build, we don't leak build-dir paths
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

# User option to enable OpenCV (defaults to OFF)
option(ENABLE_OPENCV "Enable OpenCV support for facial recognition" OFF)

# Optimization and security flags for a PAM module
add_compile_options(-O3 -pipe -fPIC -fstack-protector-strong -D_FORTIFY_SOURCE=2)

# Set the mandatory config path as a global definition
add_definitions(-DFACIALAUTH_DEFAULT_CONFIG="/etc/security/pam_facial_auth.conf")

# --- PAM CONFIGURATION (Always Required) ---
find_library(PAM_LIBRARY NAMES pam REQUIRED)
find_path(PAM_INCLUDE_DIR NAMES security/pam_modules.h REQUIRED)

# --- OPENCV CONFIGURATION (Conditional) ---
if(ENABLE_OPENCV)
    # Require only CPU components to avoid CUDA linkage
    find_package(OpenCV REQUIRED COMPONENTS
        core
        imgproc
        objdetect
        videoio
        imgcodecs
        face
    )
    add_definitions(-DHAVE_OPENCV=1)
    message(STATUS "OpenCV support enabled (CPU-only modules)")
else()
    message(STATUS "OpenCV support disabled")
endif()

# Include directories
include_directories(
    include
    ${PAM_INCLUDE_DIR}
)

if(ENABLE_OPENCV)
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()

# --- TARGETS ---

# 1. Shared library (Core logic)
add_library(facialauth SHARED src/libfacialauth.cpp)
if(ENABLE_OPENCV)
    target_link_libraries(facialauth PRIVATE ${OpenCV_LIBS})
endif()

# 2. PAM Module
add_library(pam_facial_auth MODULE src/pam_facial_auth.cpp)
set_target_properties(pam_facial_auth PROPERTIES
    PREFIX ""
    LIBRARY_OUTPUT_NAME "pam_facial_auth"
)
target_link_libraries(pam_facial_auth PRIVATE facialauth ${PAM_LIBRARY})

# 3. Helper Tools (Only built if OpenCV is enabled)
if(ENABLE_OPENCV)
    add_executable(facial_capture src/facial_capture.cpp)
    target_link_libraries(facial_capture PRIVATE facialauth ${OpenCV_LIBS})

    add_executable(facial_training src/facial_training.cpp)
    target_link_libraries(facial_training PRIVATE facialauth ${OpenCV_LIBS})

    add_executable(facial_test src/facial_test.cpp)
    target_link_libraries(facial_test PRIVATE facialauth ${OpenCV_LIBS})
endif()
