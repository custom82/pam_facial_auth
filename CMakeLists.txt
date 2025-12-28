cmake_minimum_required(VERSION 3.10)
project(pam_facial_auth LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 1. Ricerca Obbligatoria di OpenCV
# Se non viene trovata, CMake si ferma qui con un errore.
find_package(OpenCV REQUIRED COMPONENTS
    core
    imgproc
    imgcodecs
    videoio
    highgui
    objdetect
    face
    dnn
)

# 2. Directory degli header
include_directories(include ${OpenCV_INCLUDE_DIRS})

# 3. Target: Libreria principale (include i plugin)
# Compiliamo i plugin direttamente dentro la lib per semplicità di linkaggio
add_library(facialauth SHARED
    src/libfacialauth.cpp
    src/plugins/plugin_classic.cpp
    src/plugins/plugin_sface.cpp
)
target_link_libraries(facialauth PRIVATE ${OpenCV_LIBS})

# 4. Target: Modulo PAM
add_library(pam_facial_auth MODULE src/pam_facial_auth.cpp)
target_link_libraries(pam_facial_auth PRIVATE facialauth pam)
set_target_properties(pam_facial_auth PROPERTIES PREFIX "")

# 5. Target: Tool Binary
add_executable(facial_capture src/facial_capture.cpp)
target_link_libraries(facial_capture PRIVATE facialauth)

add_executable(facial_training src/facial_training.cpp)
target_link_libraries(facial_training PRIVATE facialauth)

add_executable(facial_test src/facial_test.cpp)
target_link_libraries(facial_test PRIVATE facialauth)

# 6. Installazione (Standard Gentoo/Linux)
install(TARGET_LIBRARY facialauth DESTINATION /usr/lib64)
install(TARGETS pam_facial_auth DESTINATION /lib64/security)
install(TARGETS facial_capture facial_training facial_test DESTINATION /usr/bin)
