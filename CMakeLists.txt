cmake_minimum_required(VERSION 3.10)
project(pam_facial_auth CXX)

set(CMAKE_CXX_STANDARD 20)

# Output dirs
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# PAM
find_package(PkgConfig REQUIRED)
pkg_check_modules(PAM REQUIRED pam)
include_directories(${PAM_INCLUDE_DIRS})

include_directories(include)

#
# === LIBRERIA PRINCIPALE ===
#
add_library(facialauth SHARED
    src/libfacialauth.cpp
)

target_link_libraries(facialauth ${OpenCV_LIBS})

#
# === MODULO PAM ===
#
add_library(pam_facial_auth MODULE
    src/pam_facial_auth.cpp
)

# Rimuove il prefisso "lib" → pam_facial_auth.so
set_target_properties(pam_facial_auth PROPERTIES PREFIX "")

target_link_libraries(pam_facial_auth
    facialauth
    ${PAM_LIBRARIES}
)

#
# === BINARI ===
#
add_executable(facial_capture src/facial_capture.cpp)
target_link_libraries(facial_capture facialauth ${OpenCV_LIBS})

add_executable(facial_training src/facial_training.cpp)
target_link_libraries(facial_training facialauth ${OpenCV_LIBS})

add_executable(facial_test src/facial_test.cpp)
target_link_libraries(facial_test facialauth ${OpenCV_LIBS})

#
# === INSTALLAZIONE ===
#

# PAM module
install(TARGETS pam_facial_auth
    LIBRARY DESTINATION /lib64/security
)

# Main library
install(TARGETS facialauth
    LIBRARY DESTINATION /usr/lib64
)

# CLI tools
install(TARGETS facial_capture facial_training facial_test
    RUNTIME DESTINATION /usr/sbin
)

# Headers
install(DIRECTORY include/
    DESTINATION /usr/include
)

# Install config file
if(EXISTS "${CMAKE_SOURCE_DIR}/etc/pam_facial.conf")
    install(FILES etc/pam_facial.conf
        DESTINATION /etc/security
        PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
    )
else()
    message(WARNING "pam_facial.conf non trovato, nessun file installato in /etc/security")
endif()


add_custom_target(man ALL DEPENDS
    man/facial_capture.1
    man/facial_training.1
    man/facial_test.1
    man/pam_facial_auth.8
)
