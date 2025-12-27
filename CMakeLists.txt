cmake_minimum_required(VERSION 3.10)
project(pam_facial_auth LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Trova le dipendenze
find_package(OpenCV REQUIRED COMPONENTS core imgproc face dnn imgcodecs videoio highgui objdetect)
find_package(PkgConfig REQUIRED)
pkg_check_modules(PAM REQUIRED pam)

# Directory inclusione
include_directories(include ${OpenCV_INCLUDE_DIRS} ${PAM_INCLUDE_DIRS})

# 1. Target: Libreria Core (Contiene i Plugin Classic e SFace)
# Assicurati che libfacialauth.cpp contenga le implementazioni dei plugin
add_library(facialauth SHARED
    src/libfacialauth.cpp
)
target_link_libraries(facialauth PRIVATE ${OpenCV_LIBS})

# 2. Target: Modulo PAM
add_library(pam_facial_auth MODULE src/pam_facial_auth.cpp)
set_target_properties(pam_facial_auth PROPERTIES PREFIX "")
target_link_libraries(pam_facial_auth PRIVATE facialauth ${PAM_LIBRARIES})

# 3. Target: Tool CLI
add_executable(facial_capture src/facial_capture.cpp)
target_link_libraries(facial_capture PRIVATE facialauth ${OpenCV_LIBS})

add_executable(facial_training src/facial_training.cpp)
target_link_libraries(facial_training PRIVATE facialauth ${OpenCV_LIBS})

add_executable(facial_test src/facial_test.cpp)
target_link_libraries(facial_test PRIVATE facialauth ${OpenCV_LIBS})

# Se hai file separati per i plugin (es. classic_plugin.cpp), aggiungili qui:
# target_sources(facialauth PRIVATE src/classic_plugin.cpp src/sface_plugin.cpp)
