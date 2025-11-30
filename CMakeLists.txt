cmake_minimum_required(VERSION 3.18)
project(pam_facial_auth LANGUAGES C CXX)

option(ENABLE_CUDA "Enable CUDA DNN backend support" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_SKIP_INSTALL_RPATH YES CACHE BOOL "Remove RUNPATH from ELF dynamic sections" FORCE)

#====================================================================
# OpenCV
#====================================================================
find_package(OpenCV 4.0 REQUIRED
    COMPONENTS
        core
        imgproc
        imgcodecs
        videoio
        objdetect
        dnn
        highgui
        face
)

if(ENABLE_CUDA)
    message(STATUS "CUDA support ENABLED at build-time")
    add_definitions(-DENABLE_CUDA)
else()
    message(STATUS "CUDA support DISABLED at build-time")
endif()

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

#====================================================================
# Libreria condivisa libfacialauth
#====================================================================
add_library(facialauth SHARED
    src/libfacialauth.cpp
    include/libfacialauth.h
)

target_link_libraries(facialauth
    ${OpenCV_LIBS}
)

#====================================================================
# Modulo PAM
#====================================================================
add_library(pam_facial_auth SHARED src/pam_facial_auth.cpp)

# Rimuove "lib" dal nome del modulo PAM
set_target_properties(pam_facial_auth PROPERTIES PREFIX "")

# Assicurati che venga compilato come modulo PAM
target_link_libraries(pam_facial_auth PRIVATE facialauth pam)


#====================================================================
# Eseguibili CLI reali del progetto
#====================================================================
add_executable(facial_capture  src/facial_capture.cpp)
add_executable(facial_training src/facial_training.cpp)
add_executable(facial_test     src/facial_test.cpp)

target_link_libraries(facial_capture  facialauth ${OpenCV_LIBS})
target_link_libraries(facial_training facialauth ${OpenCV_LIBS})
target_link_libraries(facial_test     facialauth ${OpenCV_LIBS})
