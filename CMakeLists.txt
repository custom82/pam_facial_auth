cmake_minimum_required(VERSION 3.10)

# Impostare il nome del progetto
project(pam_facial_auth)

# Aggiungi OpenCV e PAM come dipendenze
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(PAM REQUIRED pam)

# Aggiungi le directory di inclusione per PAM e OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PAM_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/src)  # Aggiungi la tua directory di sorgenti

# Forzare l'uso di C++20
set(CMAKE_CXX_STANDARD 20)  # Forza l'uso di C++20
set(CMAKE_CXX_STANDARD_REQUIRED ON)  # Rende obbligatorio C++20

# Aggiungi la libreria di utilità
add_library(Utils STATIC ${CMAKE_SOURCE_DIR}/src/Utils.cpp)

# Creazione del modulo PAM come libreria condivisa
add_library(pam_facial_auth MODULE
    src/pam_facial_auth.c    # Usa il file C++ modificato
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
    ${PAM_LIBRARIES}
    ${OpenCV_LIBS}
    Utils
)

# Impostare l'output della libreria
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

# Impostare l'output dei binari
set_target_properties(facial_test facial_training pam_facial_auth PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Impostare la directory di installazione per PAM
install(TARGETS pam_facial_auth
    LIBRARY DESTINATION /usr/lib64/security/
)

# Impostare la directory di installazione per i binari
install(TARGETS facial_test facial_training
    RUNTIME DESTINATION /sbin
)

# Impostare la directory di installazione per i file di configurazione
install(DIRECTORY src/config/
    DESTINATION /etc/pam_facial_auth
)
