cmake_minimum_required(VERSION 3.10)

# Impostare il nome del progetto
project(pam_facial_auth)

# Aggiungi OpenCV come dipendenza, con supporto per il modulo contrib
find_package(OpenCV REQUIRED contrib)

# Impostare C++ standard a C++20
set(CMAKE_CXX_STANDARD 20)

# Aggiungi le directory di inclusione per OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})

# Aggiungi la libreria di utilità
add_library(Utils STATIC ${CMAKE_SOURCE_DIR}/src/Utils.cpp)

# Creazione del modulo PAM come libreria condivisa
add_library(pam_facial_auth MODULE
    src/pam_facial_auth.c
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

# Impostare l'output della libreria in /usr/lib64/security
set_target_properties(pam_facial_auth PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY /usr/lib64/security/
)

# Compila il binario facial_test
add_executable(facial_test src/facial_test.cpp src/FaceRecWrapper.cpp)

# Link con le librerie necessarie per facial_test
target_link_libraries(facial_test
    PRIVATE
    ${OpenCV_LIBS}
    Utils
)

# Compila il binario facial_training
add_executable(facial_training src/facial_training.cpp src/FaceRecWrapper.cpp)

# Link con le librerie necessarie per facial_training
target_link_libraries(facial_training
    PRIVATE
    ${OpenCV_LIBS}
    Utils
)

# Impostare i binari di output
set_target_properties(facial_test facial_training pam_facial_auth PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Impostare il percorso di output della libreria
set_target_properties(facial_test pam_facial_auth PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)
