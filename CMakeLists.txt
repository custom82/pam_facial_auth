# Impostare la versione minima di CMake
cmake_minimum_required(VERSION 3.10)

# Gestire politiche per evitare conflitti di target
cmake_policy(SET CMP0002 NEW)

# Impostare il nome del progetto
project(pam_facial_auth)

# Aggiungi OpenCV e PAM come dipendenze
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(PAM REQUIRED pam)  # Usa 'pam' per pkg-config

# Aggiungi le directory di inclusione per PAM e OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PAM_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/src)  # Aggiungi la tua directory di sorgenti

# Aggiungi la libreria di utilità
add_library(Utils STATIC ${CMAKE_SOURCE_DIR}/src/Utils.cpp)

# Compila il modulo PAM (assicurati che la directory src contenga FacialAuth.cpp e altri file)
add_library(pam_facial_auth MODULE
    src/FacialAuth.cpp
    src/FaceRecWrapper.cpp
)

# Link con le librerie necessarie per pam_facial_auth
target_link_libraries(pam_facial_auth
    PRIVATE
    ${PAM_LIBRARIES}  # Usa ${PAM_LIBRARIES} per il linking di PAM
    ${OpenCV_LIBS}
    Utils
)

# Compila il binario facial_auth
add_executable(facial_auth src/facial_auth.cpp src/FaceRecWrapper.cpp)

# Link con le librerie necessarie per facial_auth
target_link_libraries(facial_auth
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

# Impostare l'output dei binari (opzionale)
set_target_properties(facial_auth facial_training pam_facial_auth PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Impostare altre configurazioni
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
