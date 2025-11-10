cmake_minimum_required(VERSION 3.10)

# Impostare il nome del progetto
project(pam_facial_auth)

# Aggiungi OpenCV come dipendenza
find_package(OpenCV REQUIRED)

# Verifica che OpenCV sia trovato
if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found!")
endif()

# Aggiungi le directory di inclusione per OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})

# Aggiungi la libreria di utilità
add_library(Utils STATIC src/Utils.cpp)

# Creazione del modulo PAM come libreria condivisa
# Rinomina pam_facial_auth.c in pam_facial_auth.cpp
add_library(pam_facial_auth MODULE
    src/pam_facial_auth.cpp  # Usa pam_facial_auth.cpp
    src/FaceRecWrapper.cpp
)

# Imposta il nome del modulo come pam_facial_auth.so
set_target_properties(pam_facial_auth PROPERTIES
    PREFIX ""
    SUFFIX ".so"
)

# Link con le librerie necessarie per PAM e OpenCV
target_link_libraries(pam_facial_auth
    PRIVATE
    ${OpenCV_LIBS}
    Utils
)

# Compilazione dei binari
add_executable(facial_test src/facial_test.cpp src/FaceRecWrapper.cpp)
target_link_libraries(facial_test
    PRIVATE
    ${OpenCV_LIBS}
    Utils
)

add_executable(facial_training src/facial_training.cpp src/FaceRecWrapper.cpp)
target_link_libraries(facial_training
    PRIVATE
    ${OpenCV_LIBS}
    Utils
)

# Impostare l'output dei binari (opzionale)
set_target_properties(facial_test facial_training pam_facial_auth PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Impostare l'output della libreria
set_target_properties(pam_facial_auth PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# Forza l'uso di C++20 per i file C++
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compila pam_facial_auth.cpp come file C++ (ora è rinominato)
set_target_properties(pam_facial_auth PROPERTIES
    COMPILE_FLAGS "-std=c++20"
)
