cmake_minimum_required(VERSION 3.10)

# Nome del progetto
project(pam_facial_auth)

# Impostazione standard C++
set(CMAKE_CXX_STANDARD 20)

# Trova OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Trova pkg-config (assicurati che sia installato)
find_package(PkgConfig REQUIRED)

# Trova PAM utilizzando pkg-config
pkg_check_modules(PAM REQUIRED pam)

# Aggiungi le directory per gli header
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${PAM_INCLUDE_DIRS}
)

# Aggiungi i file sorgente per la libreria shared (libfacialauth.so)
set(SOURCES
    src/libfacialauth.cpp
    src/facial_capture.cpp
    src/facial_training.cpp
    src/facial_test.cpp
    src/pam_facial_auth.cpp
)

# Creazione della libreria condivisa .so
add_library(facialauth SHARED ${SOURCES})

# Link alla libreria OpenCV e PAM
target_link_libraries(facialauth
    ${OpenCV_LIBS}
    ${PAM_LIBRARIES}
)

# Modulo PAM (pam_facial_auth.so)
add_library(pam_facial_auth MODULE src/pam_facial_auth.cpp)
target_link_libraries(pam_facial_auth
    facialauth  # Link al modulo PAM
    ${OpenCV_LIBS}
    ${PAM_LIBRARIES}
)

# Esegui le applicazioni di test
add_executable(facial_test src/facial_test.cpp)
target_link_libraries(facial_test facialauth ${OpenCV_LIBS})

add_executable(facial_training src/facial_training.cpp)
target_link_libraries(facial_training facialauth ${OpenCV_LIBS})

add_executable(facial_capture src/facial_capture.cpp)
target_link_libraries(facial_capture facialauth ${OpenCV_LIBS})

# Installazione del modulo PAM
install(TARGETS pam_facial_auth
    LIBRARY DESTINATION /lib/security  # Modulo PAM
)

# Installazione degli eseguibili (facial_test, facial_training, facial_capture)
install(TARGETS facial_test facial_training facial_capture
    DESTINATION /usr/local/bin
)

