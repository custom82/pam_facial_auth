cmake_minimum_required(VERSION 3.20)

project(pam_facial_auth LANGUAGES CXX)

# ======================================
# Build options
# ======================================
option(ENABLE_CUDA     "Enable CUDA support"     ON)
option(ENABLE_OPENCL   "Enable OpenCL support"   ON)

# ======================================
# Compiler flags
# ======================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")
# =========================================================
# Prevent CUDA OpenCL conflicts in RPATH
# =========================================================
set(CMAKE_SKIP_RPATH TRUE)
set(CMAKE_SKIP_BUILD_RPATH TRUE)
set(CMAKE_SKIP_INSTALL_RPATH TRUE)

# ======================================
# Locate OpenCV
# ======================================
find_package(OpenCV 4 REQUIRED COMPONENTS
    core imgproc videoio highgui dnn face
)

message(STATUS "OpenCV found: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

# ======================================
# CUDA optional detection (CMake 4 compatible)
# ======================================
if(ENABLE_CUDA)
    cmake_policy(SET CMP0146 NEW)

    find_package(CUDAToolkit)
    if(CUDAToolkit_FOUND)
        message(STATUS "CUDA support ENABLED — version ${CUDAToolkit_VERSION}")
        add_compile_definitions(ENABLE_CUDA=1)
    else()
        message(WARNING "CUDA requested but NOT detected — disabling")
        set(ENABLE_CUDA OFF)
    endif()
endif()

# ======================================
# OpenCL optional detection
# ======================================
if(ENABLE_OPENCL)
    find_package(OpenCL)
    if(OpenCL_FOUND)
        message(STATUS "OpenCL support ENABLED")
        add_compile_definitions(ENABLE_OPENCL=1)
    else()
        message(WARNING "OpenCL requested but NOT found — disabling")
        set(ENABLE_OPENCL OFF)
    endif()
endif()

# ======================================
# Include paths
# ======================================
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)

# ======================================
# Sources
# ======================================
set(SOURCES_LIB
    src/libfacialauth.cpp
)

set(SOURCES_PAM
    src/pam_facial_auth.cpp
)

set(SOURCES_CAPTURE
    src/facial_capture.cpp
)

set(SOURCES_TRAIN
    src/facial_training.cpp
)

set(SOURCES_TEST
    src/facial_test.cpp
)

# ======================================
# Build shared library: libfacialauth.so
# ======================================
add_library(facialauth SHARED ${SOURCES_LIB})
target_link_libraries(facialauth ${OpenCV_LIBS})

if(ENABLE_OPENCL AND OpenCL_FOUND)
    target_link_libraries(facialauth OpenCL::OpenCL)
endif()

if(ENABLE_CUDA AND CUDAToolkit_FOUND)
    target_link_libraries(facialauth CUDA::cudart CUDA::cuda_driver)
endif()

# ======================================
# Build PAM module
# ======================================
add_library(pam_facial_auth SHARED ${SOURCES_PAM})
# Rimuovere il prefisso lib e chiamarlo pam_facial_auth.so
set_target_properties(pam_facial_auth PROPERTIES
    PREFIX ""                # niente "lib"
    OUTPUT_NAME "pam_facial_auth"  # nome del modulo PAM
)
# ======================================
# Build CLI tools
# ======================================
add_executable(facial_capture  ${SOURCES_CAPTURE})
add_executable(facial_training ${SOURCES_TRAIN})
add_executable(facial_test     ${SOURCES_TEST})

target_link_libraries(facial_capture  facialauth)
target_link_libraries(facial_training facialauth)
target_link_libraries(facial_test     facialauth)

# ======================================
# Build Summary Output
# ======================================
message(STATUS "======== Build summary ========")
message(STATUS "CUDA enabled: ${ENABLE_CUDA}")
message(STATUS "OpenCL enabled: ${ENABLE_OPENCL}")
message(STATUS "===============================")
