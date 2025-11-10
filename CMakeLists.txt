# CMake minimum version
cmake_minimum_required(VERSION 3.10)

# Project definition
project(pam_facial_auth VERSION 1.0)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the installation prefix
set(CMAKE_INSTALL_PREFIX /usr)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find PAM library (set it manually if it's in a non-standard location)
find_library(PAM_LIBRARIES NAMES pam REQUIRED)

if(NOT PAM_LIBRARIES)
  message(FATAL_ERROR "Could not find the PAM library. Please install PAM or specify the correct path.")
endif()

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    /usr/include/pam
)

# Sources
set(SOURCES
    src/FacialAuth.cpp
    src/FaceRecWrapper.cpp
    facial_test.cpp
)

# Headers
set(HEADERS
    src/FacialAuth.h
    src/FaceRecWrapper.h
)

# Define the executable
add_executable(pam_facial_auth ${SOURCES} ${HEADERS})

# Link libraries
target_link_libraries(pam_facial_auth
    ${OpenCV_LIBS}
    ${PAM_LIBRARIES}
)

# Install target
install(TARGETS pam_facial_auth DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

# Add tests if necessary
enable_testing()
add_test(NAME FacialAuthTest COMMAND pam_facial_auth)
