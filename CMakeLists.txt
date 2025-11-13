cmake_minimum_required(VERSION 3.10)

# Nome del progetto
project(pam_facial_auth)

# Impostazione standard C++
set(CMAKE_CXX_STANDARD 20)

# Trova OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Usa pkg-config per trovare PAM
find_package(PkgConfig REQUIRED)
pkg_check_modules(PAM REQUIRED pam)

# Aggiungi le directory per gli header
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${PAM_INCLUDE_DIRS}  # Include i path di PAM da pkg-config
)

# Aggiungi i file sorgente per la libreria shared (libfacialauth.so)
set(SOURCES
    src/libfacialauth.cpp
)

# Creazione della libreria condivisa .so
add_library(facialauth SHARED ${SOURCES})

# Link alla libreria OpenCV e PAM
target_link_libraries(facialauth
    ${OpenCV_LIBS}
    ${PAM_LIBRARIES}  # Link con le librerie PAM
)

# Aggiungi eseguibili separati per facial_capture, facial_training e facial_test
add_executable(facial_capture src/facial_capture.cpp)
target_link_libraries(facial_capture
    facialauth  # Link alla libreria shared facialauth (già include OpenCV)
)

add_executable(facial_training src/facial_training.cpp)
target_link_libraries(facial_training
    facialauth  # Link alla libreria shared facialauth (già include OpenCV)
)

add_executable(facial_test src/facial_test.cpp)
target_link_libraries(facial_test
    facialauth  # Link alla libreria shared facialauth (già include OpenCV)
)

# Modulo PAM (pam_facial_auth.so)
add_library(pam_facial_auth MODULE src/pam_facial_auth.cpp)
target_link_libraries(pam_facial_auth
    facialauth  # Link al modulo PAM (che dipende da facialauth)
    ${PAM_LIBRARIES}  # Link con le librerie PAM
)

# Installazione del modulo PAM
install(TARGETS pam_facial_auth
    LIBRARY DESTINATION /lib64/security  # Modulo PAM
)
