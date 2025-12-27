cmake_minimum_required(VERSION 3.22)
project(pam_facial_auth LANGUAGES CXX)

# --- STANDARD & COMPILER CONFIG ---
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Gentoo QA Fix: Remove insecure RUNPATHs (RPATH/RUNPATH)
# This prevents leaking build-time paths (like Portage temp dirs or CUDA paths)
set(CMAKE_SKIP_RPATH TRUE)
set(CMAKE_SKIP_INSTALL_RPATH TRUE)

# --- OPTIONS ---
option(ENABLE_OPENCV "Enable OpenCV support for facial recognition" OFF)

# Global definitions
add_definitions(-DFACIALAUTH_DEFAULT_CONFIG="/etc/security/pam_facial_auth.conf")

# Security-hardened compiler options
add_compile_options(-O3 -pipe -fPIC -fstack-protector-strong -D_FORTIFY_SOURCE=2)

# --- DEPENDENCIES ---

# PAM is mandatory for a PAM module
find_library(PAM_LIBRARY NAMES pam REQUIRED)
find_path(PAM_INCLUDE_DIR NAMES security/pam_modules.h REQUIRED)

# OpenCV is conditional to keep the build clean
if(ENABLE_OPENCV)
    # We explicitly request only CPU components to avoid CUDA linkage inheritance
    find_package(OpenCV REQUIRED COMPONENTS
        core
        imgproc
        objdetect
        videoio
        imgcodecs
        face
    )
    message(STATUS "OpenCV support enabled (CPU-only modules)")
endif()

# --- TARGETS ---

# 1. Core Library (facialauth)
# This library contains the logic for config parsing and OpenCV wrappers
add_library(facialauth SHARED src/libfacialauth.cpp)

# Public include directories (propagated to anyone linking this library)
target_include_directories(facialauth PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${PAM_INCLUDE_DIR}
)

if(ENABLE_OPENCV)
    # This define is critical: it enables the #ifdef HAVE_OPENCV blocks in the code
    target_compile_definitions(facialauth PUBLIC HAVE_OPENCV=1)
    target_include_directories(facialauth PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(facialauth PUBLIC ${OpenCV_LIBS})
endif()

# 2. PAM Module (pam_facial_auth)
# Must be a MODULE for PAM to load it via dlopen.
# PREFIX "" prevents it from being named 'libpam_facial_auth.so'
add_library(pam_facial_auth MODULE src/pam_facial_auth.cpp)
set_target_properties(pam_facial_auth PROPERTIES
    PREFIX ""
    LIBRARY_OUTPUT_NAME "pam_facial_auth"
)
target_link_libraries(pam_facial_auth PRIVATE facialauth ${PAM_LIBRARY})

# 3. Helper Tools (Built only if OpenCV is enabled)
if(ENABLE_OPENCV)
    # Tool for capturing faces
    add_executable(facial_capture src/facial_capture.cpp)
    target_link_libraries(facial_capture PRIVATE facialauth)

    # Tool for training the model
    add_executable(facial_training src/facial_training.cpp)
    target_link_libraries(facial_training PRIVATE facialauth)

    # Tool for testing recognition
    add_executable(facial_test src/facial_test.cpp)
    target_link_libraries(facial_test PRIVATE facialauth)
endif()
