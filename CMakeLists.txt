cmake_minimum_required(VERSION 3.10)

project(PAM_Facial_Auth)

# Aggiungi i percorsi per le librerie e i file di intestazione di OpenCV e PAM
find_package(OpenCV REQUIRED)
find_package(PAM REQUIRED)

# Impostare la directory di inclusione per il progetto
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${PAM_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include  # Per includere il file pam_facial_auth.h
)

# Aggiungi i file sorgente
set(SOURCES
    src/pam_facial_auth.cpp
    src/facial_test.cpp
    src/facial_training.cpp
    src/FacialAuth.cpp
    src/FaceRecWrapper.cpp
)

# Creazione della libreria del modulo PAM
add_library(pam_facial_auth MODULE ${SOURCES})

# Linkare OpenCV e PAM alla libreria del modulo
target_link_libraries(pam_facial_auth ${OpenCV_LIBS} ${PAM_LIBRARIES})

# Creazione degli eseguibili per il test e il training
add_executable(facial_test src/facial_test.cpp)
target_link_libraries(facial_test ${OpenCV_LIBS})

add_executable(facial_training src/facial_training.cpp)
target_link_libraries(facial_training ${OpenCV_LIBS})

# Per la compilazione con debug informazioni
set(CMAKE_BUILD_TYPE Debug)

