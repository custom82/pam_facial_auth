cmake_minimum_required(VERSION 3.10)

# Nome del progetto
project(PAM_Facial_Auth)

# Impostare il linguaggio C++
set(CMAKE_CXX_STANDARD 20)

# Trova OpenCV
find_package(OpenCV REQUIRED)

# Trova PAM (Personal Authentication Module)
find_package(PAM REQUIRED)

# Aggiungi la directory src
add_subdirectory(src)

# Aggiungi l'eseguibile di test
add_executable(facial_test src/facial_test.cpp)

# Collegamento con le librerie di OpenCV e PAM
target_link_libraries(facial_test
    ${OpenCV_LIBS}  # Collegamento con OpenCV
    ${PAM_LIBRARIES}  # Collegamento con PAM
)

# Aggiungere le librerie di PAM Faciale
add_library(pam_facial_auth SHARED
    src/FacialAuth.cpp
    src/FaceRecWrapper.cpp
)

# Collegamento delle librerie OpenCV e PAM alla libreria pam_facial_auth
target_link_libraries(pam_facial_auth
    ${OpenCV_LIBS}  # Collegamento con OpenCV
    ${PAM_LIBRARIES}  # Collegamento con PAM
)

# Include delle directory OpenCV
include_directories(${O_
