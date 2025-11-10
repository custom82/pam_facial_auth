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

# Sources for pam_facial_auth
set(PAM_FACIAL_AUTH_SOURCES
    src/FacialAuth.cpp
    src/FaceRecWrapper.cpp
)

# Sources for facial_test
set(FACIAL_TEST_SOURCES
    facial_test.cpp
    src/FaceRecWrapper.cpp
)

# Sources for facial_training
set(FACIAL_TRAINING_SOURCES
    facial_training.cpp
    src/FaceRecWrapper.cpp
)

# Headers
set(HEADERS
    src/FacialAuth.h
    src/FaceRecWrapper.h
)

# Define the executable for pam_facial_auth
add_executable(pam_facial_auth ${PAM_FACIAL_AUTH_SOURCES} ${HEADERS})

# Define the executable for facial_test
add_executable(facial_test ${FACIAL_TEST_SOURCES} ${HEADERS})

# Define the executable for facial_training
add_executable(facial_training ${FACIAL_TRAINING_SOURCES} ${HEADERS})

# Link libraries for pam_facial_auth
target_link_libraries(pam_facial_auth
    ${OpenCV_LIBS}
    ${PAM_LIBRARIES}
)

# Link libraries for facial_test
target_link_libraries(facial_test
    ${OpenCV_LIBS}
    ${PAM_LIBRARIES}
)

# Link libraries for facial_training
target_link_libraries(facial_training
    ${OpenCV_LIBS}
    ${PAM_LIBRARIES}
)

# Install targets
install(TARGETS pam_facial_auth facial_test facial_training DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

# Add tests if necessary
enable_testing()
add_test(NAME FacialAuthTest COMMAND pam_facial_auth)
