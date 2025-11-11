cmake_minimum_required(VERSION 3.10)

# Impostare il nome del progetto
project(pam_facial_auth)

# Aggiungi OpenCV come dipendenza
find_package(OpenCV REQUIRED)

# Verifica che OpenCV sia stato trovato
if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found!")
endif()

# Aggiungi le directory di inclusione per OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})

# Aggiungi la libreria di utilità
add_library(Utils STATIC src/Utils.cpp)

# =========================
# Modulo PAM principale
# =========================
add_library(pam_facial_auth MODULE
    src/pam_facial_auth.cpp
    src/FaceRecWrapper.cpp
)

# Imposta il nome del modulo come pam_facial_auth.so
set_target_properties(pam_facial_auth PROPERTIES
    PREFIX ""
    SUFFIX ".so"
)

# Link con le librerie necessarie
target_link_libraries(pam_facial_auth
    PRIVATE
    ${OpenCV_LIBS}
    Utils
    pam  # Aggiungi il linking con libpam
)

# =========================
# Tool: facial_test
# =========================
add_executable(facial_test
    src/facial_test.cpp
    src/FaceRecWrapper.cpp
)

target_link_libraries(facial_test
    PRIVATE
    ${OpenCV_LIBS}
    Utils
)

# =========================
# Tool: facial_training
# =========================
add_executable(facial_training
    src/facial_training.cpp
    src/FaceRecWrapper.cpp
)

target_link_libraries(facial_training
    PRIVATE
    ${OpenCV_LIBS}
    Utils
)

# =========================
# Tool: facial_capture
# =========================
add_executable(facial_capture
    src/facial_capture.cpp
)

target_link_libraries(facial_capture
    PRIVATE
    ${OpenCV_LIBS}
    Utils
)

# =========================
# Impostazioni di output
# =========================
set_target_properties(facial_test facial_training facial_capture pam_facial_auth PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

set_target_properties(pam_facial_auth PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# =========================
# Impostazioni di compilazione
# =========================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Forza la compilazione C++20 per il modulo PAM
set_target_properties(pam_facial_auth PROPERTIES
    COMPILE_FLAGS "-std=c++20"
)
