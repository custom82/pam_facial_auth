cmake_minimum_required(VERSION 4.2)
project(pam_facial_auth
    VERSION 1.1.0
    LANGUAGES CXX
)

# --- Standard C++20 ---
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- Gestione Dipendenze ---

# PAM: Ricerca manuale per evitare errori di configurazione su Gentoo
find_library(PAM_LIBRARY NAMES pam REQUIRED)
find_path(PAM_INCLUDE_DIR NAMES security/pam_appl.h REQUIRED)

# OpenCV: Componenti necessari
find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio imgcodecs dnn objdetect face)

# --- Target: Libreria Core (libfacialauth) ---

add_library(facialauth SHARED)

target_sources(facialauth PRIVATE
    src/libfacialauth.cpp
)

target_include_directories(facialauth
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${PAM_INCLUDE_DIR}
)

target_link_libraries(facialauth
    PUBLIC
        ${OpenCV_LIBS}
    PRIVATE
        ${PAM_LIBRARY}
)

# --- Target: Modulo PAM (pam_facial_auth.so) ---

# MODULE è perfetto per i plugin caricati via dlopen (come i moduli PAM)
add_library(pam_facial_auth MODULE)

target_sources(pam_facial_auth PRIVATE
    src/pam_facial_auth.cpp
)

target_link_libraries(pam_facial_auth
    PRIVATE
        facialauth
        ${PAM_LIBRARY}
)

# Rimuoviamo il prefisso "lib" per compatibilità con lo stack PAM
set_target_properties(pam_facial_auth PROPERTIES
    PREFIX ""
)

# --- Target: Utility (Eseguibili) ---

set(TOOLS facial_test facial_training facial_capture)

foreach(tool IN LISTS TOOLS)
    add_executable(${tool} src/${tool}.cpp)
    target_link_libraries(${tool} PRIVATE facialauth)
endforeach()
