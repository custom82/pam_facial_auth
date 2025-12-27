cmake_minimum_required(VERSION 3.22)
project(pam_facial_auth LANGUAGES CXX)

# Standard C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Gentoo QA: Rimuove i RUNPATH insicuri (evita leak di CUDA o path di build)
set(CMAKE_SKIP_RPATH TRUE)
set(CMAKE_SKIP_INSTALL_RPATH TRUE)

# Opzioni di sicurezza e ottimizzazione
add_compile_options(-O3 -pipe -fPIC -fstack-protector-strong -D_FORTIFY_SOURCE=2)
add_definitions(-DFACIALAUTH_DEFAULT_CONFIG="/etc/security/pam_facial_auth.conf")

# --- DIPENDENZE OBBLIGATORIE ---

# PAM
find_library(PAM_LIBRARY NAMES pam REQUIRED)
find_path(PAM_INCLUDE_DIR NAMES security/pam_modules.h REQUIRED)

# OpenCV (Necessario per il funzionamento biometrico e i tool)
find_package(OpenCV REQUIRED COMPONENTS
    core
    imgproc
    objdetect
    videoio
    imgcodecs
    face
)

# Definiamo HAVE_OPENCV per compatibilità con il codice sorgente esistente
add_definitions(-DHAVE_OPENCV=1)

# --- TARGETS ---

# 1. Libreria Core
add_library(facialauth SHARED src/libfacialauth.cpp)
target_include_directories(facialauth PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${PAM_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(facialauth PUBLIC ${OpenCV_LIBS})

# 2. Modulo PAM
add_library(pam_facial_auth MODULE src/pam_facial_auth.cpp)
set_target_properties(pam_facial_auth PROPERTIES
    PREFIX ""
    LIBRARY_OUTPUT_NAME "pam_facial_auth"
)
target_link_libraries(pam_facial_auth PRIVATE facialauth ${PAM_LIBRARY})

# 3. Helper Tools (Ora sempre compilati)
add_executable(facial_capture src/facial_capture.cpp)
target_link_libraries(facial_capture PRIVATE facialauth)

add_executable(facial_training src/facial_training.cpp)
target_link_libraries(facial_training PRIVATE facialauth)

add_executable(facial_test src/facial_test.cpp)
target_link_libraries(facial_test PRIVATE facialauth)
