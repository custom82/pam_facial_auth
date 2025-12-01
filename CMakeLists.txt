cmake_minimum_required(VERSION 3.10)
project(pam_facial_auth LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Opzioni di build
option(ENABLE_CUDA   "Enable CUDA support"   OFF)
option(ENABLE_OPENCL "Enable OpenCL support" OFF)

# Messaggi e macro di compilazione
if(ENABLE_CUDA)
    message(STATUS "CUDA support ENABLED at build-time")
    add_definitions(-DENABLE_CUDA)
else()
    message(STATUS "CUDA support DISABLED at build-time")
endif()

if(ENABLE_OPENCL)
    message(STATUS "OpenCL support ENABLED at build-time")
    add_definitions(-DENABLE_OPENCL)
else()
    message(STATUS "OpenCL support DISABLED at build-time")
endif()

# OpenCV
find_package(OpenCV 4.0 REQUIRED
    COMPONENTS core imgproc imgcodecs videoio objdetect dnn highgui face
)

find_package(Threads REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Libreria principale
add_library(facialauth SHARED
    src/libfacialauth.cpp
)

target_link_libraries(facialauth
    ${OpenCV_LIBS}
    Threads::Threads
)

# Modulo PAM
add_library(pam_facial_auth SHARED
    src/pam_facial_auth.cpp
)

target_link_libraries(pam_facial_auth
    facialauth
    ${OpenCV_LIBS}
    Threads::Threads
)

# Nome corretto del modulo PAM (senza "lib" davanti)
set_target_properties(pam_facial_auth PROPERTIES
    PREFIX ""
    OUTPUT_NAME "pam_facial_auth"
)

# Tool CLI
add_executable(facial_capture
    src/facial_capture.cpp
)

target_link_libraries(facial_capture
    facialauth
    ${OpenCV_LIBS}
    Threads::Threads
)

add_executable(facial_training
    src/facial_training.cpp
)

target_link_libraries(facial_training
    facialauth
    ${OpenCV_LIBS}
    Threads::Threads
)

add_executable(facial_test
    src/facial_test.cpp
)

target_link_libraries(facial_test
    facialauth
    ${OpenCV_LIBS}
    Threads::Threads
)
