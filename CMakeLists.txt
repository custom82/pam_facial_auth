# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.10)

# Set the project name and version
project(PAM_Facial_Auth VERSION 1.0)

# Set C++ standard to C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include directories (update with your paths)
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(/usr/include/opencv4)  # Assuming OpenCV is installed at this path
include_directories(/usr/include/security)  # Include PAM headers

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Find the PAM library
find_package(PAM REQUIRED)

# Add source files for the library and executables
set(SOURCE_FILES
    src/pam_facial_auth.cpp
    src/facial_capture.cpp
    src/facial_training.cpp
    src/facial_test.cpp
    src/FaceRecWrapper.cpp
    src/FacialAuth.cpp
    src/Utils.cpp
)

# Build the PAM module shared library (libpam_facial_auth.so)
add_library(pam_facial_auth SHARED ${SOURCE_FILES})

# Link with OpenCV and PAM
target_link_libraries(pam_facial_auth ${OpenCV_LIBS} ${PAM_LIBRARIES})

# Add the facial_test executable
add_executable(facial_test src/facial_test.cpp)
target_link_libraries(facial_test pam_facial_auth ${OpenCV_LIBS} ${PAM_LIBRARIES})

# Add the facial_training executable
add_executable(facial_training src/facial_training.cpp)
target_link_libraries(facial_training pam_facial_auth ${OpenCV_LIBS} ${PAM_LIBRARIES})

# Set PAM-specific flags for the module
set_target_properties(pam_facial_auth PROPERTIES PREFIX "lib" SUFFIX ".so")

# Optionally, set the installation directories for your module and executables
install(TARGETS pam_facial_auth
    LIBRARY DESTINATION /lib/security  # This is where PAM expects the module to go
)

install(TARGETS facial_test facial_training
    RUNTIME DESTINATION /usr/local/bin
)
