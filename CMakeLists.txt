# Progetto
cmake_minimum_required(VERSION 3.10)
project(pam_facial_auth VERSION 1.0 LANGUAGES CXX)

# Imposta C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Aggiungi la directory principale come directory di inclusione
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(/usr/include/opencv4)  # Aggiungi il percorso per OpenCV

# Impostazioni per PAM
find_package(PAM REQUIRED)  # Aggiungi il pacchetto PAM
include_directories(${PAM_INCLUDE_DIRS})  # Aggiungi le directory di inclusione PAM

# Aggiungi la libreria OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Aggiungi i file sorgenti
set(SOURCES
    src/pam_facial_auth.cpp
    src/facial_capture.cpp
    src/facial_training.cpp
    src/facial_test.cpp
    src/FaceRecWrapper.cpp
    src/FacialAuth.cpp
    src/Utils.cpp
)

# Aggiungi l'eseguibile (se necessario) o la libreria del modulo PAM
add_library(pam_facial_auth MODULE ${SOURCES})

# Collegamento delle librerie necessarie (OpenCV, PAM, etc.)
target_link_libraries(pam_facial_auth ${PAM_LIBRARIES} ${OpenCV_LIBS})

# Impostazioni di compilazione
set_target_properties(pam_facial_auth PROPERTIES
    PREFIX "pam_"  # Nome del modulo PAM
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# Definisci la configurazione per l'installazione (opzionale)
install(TARGETS pam_facial_auth
    LIBRARY DESTINATION /lib/security  # Directory di destinazione per i moduli PAM
)

# Opzioni di debug (opzionale)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(pam_facial_auth PRIVATE -g -Wall)
endif()
