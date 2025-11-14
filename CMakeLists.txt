cmake_minimum_required(VERSION 3.10)

project(pam_facial_auth LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV
find_package(OpenCV REQUIRED)

# PAM via pkg-config
find_package(PkgConfig REQUIRED)
pkg_check_modules(PAM REQUIRED pam)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${PAM_INCLUDE_DIRS}
)

# Libreria condivisa con tutta la logica OpenCV
set(FACIALAUTH_SOURCES
    src/libfacialauth.cpp
)

add_library(facialauth SHARED ${FACIALAUTH_SOURCES})

target_link_libraries(facialauth
    ${OpenCV_LIBS}
    ${PAM_LIBRARIES}
)

# Eseguibili (linkano SOLO a facialauth)
add_executable(facial_capture src/facial_capture.cpp)
target_link_libraries(facial_capture facialauth)

add_executable(facial_training src/facial_training.cpp)
target_link_libraries(facial_training facialauth)

add_executable(facial_test src/facial_test.cpp)
target_link_libraries(facial_test facialauth)

# Modulo PAM (C++ ma con symbol C per PAM)
add_library(pam_facial_auth MODULE src/pam_facial_auth.cpp)
target_link_libraries(pam_facial_auth
    facialauth
    ${PAM_LIBRARIES}
)

# Install: modulo PAM
install(TARGETS pam_facial_auth
    LIBRARY DESTINATION /lib64/security
)

# (eventuale) install dei binari
install(TARGETS facial_capture facial_training facial_test
    RUNTIME DESTINATION /usr/bin
)
