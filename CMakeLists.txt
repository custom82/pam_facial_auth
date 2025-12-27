cmake_minimum_required(VERSION 3.16)
project(pam_facial_auth VERSION 1.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# --- Ricerca Dipendenze ---
# Cerchiamo OpenCV con tutti i moduli necessari per i plugin (DNN e Face)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio face objdetect dnn)

# La libreria PAM è fondamentale per il modulo di autenticazione
find_library(PAM_LIBRARY NAMES pam REQUIRED)

# --- Configurazione Directory ---
include_directories(${CMAKE_SOURCE_DIR}/include)
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

# --- 1. Libreria Core (libfacialauth) ---
# Compiliamo la logica core che contiene la Factory e i Plugin
add_library(facialauth SHARED
    src/libfacialauth.cpp
)
target_link_libraries(facialauth PRIVATE ${OpenCV_LIBS})

# --- 2. Modulo PAM (pam_facial_auth.so) ---
# Deve essere un modulo caricabile dinamicamente senza prefisso "lib"
add_library(pam_facial_auth MODULE
    src/pam_facial_auth.cpp
)
set_target_properties(pam_facial_auth PROPERTIES PREFIX "")
target_link_libraries(pam_facial_auth PRIVATE facialauth ${PAM_LIBRARY})

# --- 3. Tools CLI ---
# Tool per la cattura immagini (facial_capture)
add_executable(facial_capture src/facial_capture.cpp)
target_link_libraries(facial_capture PRIVATE facialauth ${OpenCV_LIBS})

# Tool per l'addestramento (facial_training)
add_executable(facial_training src/facial_training.cpp)
target_link_libraries(facial_training PRIVATE facialauth ${OpenCV_LIBS})

# Tool per il test (facial_test)
add_executable(facial_test src/facial_test.cpp)
target_link_libraries(facial_test PRIVATE facialauth ${OpenCV_LIBS})

# --- 4. Installazione ---
# Definiamo dove vanno i file nel sistema Linux
install(TARGETS pam_facial_auth DESTINATION /lib/security)
install(TARGETS facial_capture facial_training facial_test DESTINATION /usr/local/bin)
install(FILES pam_facial.conf DESTINATION /etc/security)

# --- 5. Generazione Tarball (CPack) ---
set(CPACK_GENERATOR "TGZ")
set(CPACK_PACKAGE_NAME "pam-facial-auth-plugin")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
include(CPack)
