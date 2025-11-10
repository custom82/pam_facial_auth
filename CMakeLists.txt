cmake_minimum_required(VERSION 3.10)

project(pam_facial_auth)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)

# Aggiungi le directory di inclusione
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${OpenCV_INCLUDE_DIRS})

# Aggiungi i file sorgenti
add_executable(facial_test facial_test.cpp)
add_library(pam_facial_auth SHARED src/FacialAuth.cpp src/FaceRecWrapper.cpp)

# Collega OpenCV e PAM
find_package(OpenCV REQUIRED)
find_library(PAM_LIBRARIES REQUIRED)

# Link con le librerie
target_link_libraries(facial_test ${OpenCV_LIBS} ${PAM_LIBRARIES})
target_link_libraries(pam_facial_auth ${OpenCV_LIBS} ${PAM_LIBRARIES})

