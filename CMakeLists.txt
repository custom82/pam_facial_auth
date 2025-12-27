cmake_minimum_required(VERSION 3.22)
project(pam_facial_auth LANGUAGES CXX)

# Force C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Fix Gentoo QA Notice: "Insecure RUNPATHs"
# This prevents hardcoded build paths and CUDA paths from leaking into binaries
set(CMAKE_SKIP_RPATH TRUE)
set(CMAKE_SKIP_INSTALL_RPATH TRUE)

# Build Options
option(ENABLE_OPENCV "Enable OpenCV support for facial recognition" OFF)

# Hardcoded security configuration path (The only allowed hardcoded path)
add_definitions(-DFACIALAUTH_DEFAULT_CONFIG="/etc/security/pam_facial_auth.conf")

# Security and Optimization flags
add_compile_options(-O3 -pipe -fPIC -fstack-protector-strong -D_FORTIFY_SOURCE=2)

# --- DEPENDENCIES ---

# PAM is always required
find_library(PAM_LIBRARY NAMES pam REQUIRED)
find_path(PAM_INCLUDE_DIR NAMES security/pam_modules.h REQUIRED)

# OpenCV is conditional to avoid heavy dependencies when not needed
if(ENABLE_OPENCV)
    # Require only CPU components to explicitly avoid CUDA link inheritance
    find_package(OpenCV REQUIRED COMPONENTS
        core
        imgproc
        objdetect
        videoio
        imgcodecs
        face
    )
    add_definitions(-DHAVE_OPENCV=1)
    message(STATUS "OpenCV support enabled: Using CPU components")
endif()

# --- TARGETS ---

# 1. Core Library
add_library(facialauth SHARED src/libfacialauth.cpp)

# Public includes: headers in 'include' and PAM headers
target_include_directories(facialauth PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${PAM_INCLUDE_DIR}
)

if(ENABLE_OPENCV)
    target_include_directories(facialauth PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(facialauth PUBLIC ${OpenCV_LIBS})
endif()

# 2. PAM Module
# PREFIX "" ensures the file is 'pam_facial_auth.so' and not 'libpam_facial_auth.so'
add_library(pam_facial_auth MODULE src/pam_facial_auth.cpp)
set_target_properties(pam_facial_auth PROPERTIES
    PREFIX ""
    LIBRARY_OUTPUT_NAME "pam_facial_auth"
)
target_link_libraries(pam_facial_auth PRIVATE facialauth ${PAM_LIBRARY})

# 3. Helper Binaries (Built only if OpenCV is enabled)
if(ENABLE_OPENCV)
    add_executable(facial_capture src/facial_capture.cpp)
    target_link_libraries(facial_capture PRIVATE facialauth)

    add_executable(facial_training src/facial_training.cpp)
    target_link_libraries(facial_training PRIVATE facialauth)

    add_executable(facial_test src/facial_test.cpp)
    target_link_libraries(facial_test PRIVATE facialauth)
endif()
