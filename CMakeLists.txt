cmake_minimum_required(VERSION 3.10)

project(pam_facial_auth)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ======================================================
# TROVA OPENCV (solo per la libreria libfacialauth)
# ======================================================
find_package(OpenCV REQUIRED)

# ======================================================
# DIRECTORY PER GLI HEADER
# ======================================================
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

# ======================================================
# LIBRERIA condivisa: libfacialauth.so
# (contiene TUTTO: detection + LBPH + camera)
# ======================================================
add_library(facialauth SHARED
    src/libfacialauth.cpp
    include/libfacialauth.h
)

target_link_libraries(facialauth
    ${OpenCV_LIBS}     # SOLO QUI
)

target_include_directories(facialauth PUBLIC include)

# ======================================================
# BINARI: NON devono linkare OpenCV
# SOLO la libreria
# ======================================================

# ---- facial_capture ----
add_executable(facial_capture
    src/facial_capture.cpp
)

target_link_libraries(facial_capture
    facialauth         # SOLO libreria
)

target_include_directories(facial_capture PUBLIC include)

# ---- facial_training ----
add_executable(facial_training
    src/facial_training.cpp
)

target_link_libraries(facial_training
    facialauth
)

target_include_directories(facial_training PUBLIC include)

# ---- facial_test ----
add_executable(facial_test
    src/facial_test.cpp
)

target_link_libraries(facial_test
    facialauth
)

target_include_directories(facial_test PUBLIC include)

# ======================================================
# Modulo PAM
# ======================================================
add_library(pam_facial_auth MODULE src/pam_facial_auth.cpp)

target_link_libraries(pam_facial_auth
    facialauth
    pam               # PAM, NON OpenCV
)

install(TARGETS pam_facial_auth
    LIBRARY DESTINATION /lib64/security
)

