cmake_minimum_required(VERSION 3.20)
project(pam_facial_auth LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(ENABLE_CUDA "Enable CUDA acceleration" OFF)
option(ENABLE_OPENCL "Enable OpenCL acceleration" OFF)

find_package(OpenCV REQUIRED core imgproc objdetect videoio imgcodecs dnn face)
find_package(PkgConfig REQUIRED)

if(ENABLE_CUDA)
    find_package(CUDA REQUIRED)
endif()

if(ENABLE_OPENCL)
    find_package(OpenCL REQUIRED)
endif()

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${OpenCL_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ============================================================
# Shared library libfacialauth.so
# ============================================================
add_library(facialauth SHARED
    src/libfacialauth.cpp
)

target_link_libraries(facialauth
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${OpenCL_LIBRARIES}
    dl
    rt
)

target_compile_definitions(facialauth PRIVATE
    $<$<BOOL:${ENABLE_CUDA}>:ENABLE_CUDA>
    $<$<BOOL:${ENABLE_OPENCL}>:ENABLE_OPENCL>
)

# ============================================================
# PAM module pam_facial_auth.so
# ============================================================
add_library(pam_facial_auth SHARED
    src/pam_facial_auth.cpp
)

target_link_libraries(pam_facial_auth
    facialauth
    pam
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${OpenCL_LIBRARIES}
)

# PAM MODULE MUST NOT HAVE MAIN()
set_target_properties(pam_facial_auth PROPERTIES PREFIX "")

# ============================================================
# Executables (these REQUIRE main())
# ============================================================

add_executable(facial_capture src/facial_capture.cpp)
target_link_libraries(facial_capture facialauth ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${OpenCL_LIBRARIES})

add_executable(facial_training src/facial_training.cpp)
target_link_libraries(facial_training facialauth ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${OpenCL_LIBRARIES})

add_executable(facial_test src/facial_test.cpp)
target_link_libraries(facial_test facialauth ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${OpenCL_LIBRARIES})

# ============================================================
# Install
# ============================================================
install(TARGETS pam_facial_auth DESTINATION lib64/security)
install(TARGETS facial_capture facial_training facial_test RUNTIME DESTINATION sbin)
install(TARGETS facialauth LIBRARY DESTINATION lib64)
install(FILES etc/pam_facial.conf DESTINATION /etc/security)
