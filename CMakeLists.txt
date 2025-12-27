cmake_minimum_required(VERSION 3.16)
project(pam_facial_auth VERSION 1.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# --- 1. Ricerca Dipendenze ---
# Nota: Su Gentoo/Portage è fondamentale che OpenCV sia compilato con i moduli corretti
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio face objdetect dnn)

# Cerchiamo la libreria PAM
find_library(PAM_LIBRARY NAMES pam REQUIRED)

# --- 2. Configurazione Directory ---
# Aggiungiamo la cartella include del progetto
include_directories(${CMAKE_SOURCE_DIR}/include)

# --- 3. Target: Libreria Core (facialauth) ---
add_library(facialauth SHARED src/libfacialauth.cpp)
# Questo comando "attacca" OpenCV alla libreria e ne propaga gli header (include)
target_link_libraries(facialauth PUBLIC ${OpenCV_LIBS})

# --- 4. Target: Modulo PAM (pam_facial_auth) ---
# Usiamo MODULE perché è un plugin caricato da PAM, non una libreria linkata
add_library(pam_facial_auth MODULE src/pam_facial_auth.cpp)
set_target_properties(pam_facial_auth PROPERTIES PREFIX "")

# Linkiamo facialauth (che a sua volta porta con sé OpenCV) e PAM
target_link_libraries(pam_facial_auth PRIVATE facialauth ${PAM_LIBRARY})

# --- 5. Target: Tools CLI ---
foreach(TOOL facial_capture facial_training facial_test)
    add_executable(${TOOL} src/${TOOL}.cpp)
    target_link_libraries(${TOOL} PRIVATE facialauth)
endforeach()

# --- 6. Installazione ---
install(TARGETS pam_facial_auth DESTINATION /lib64/security) # Su Gentoo spesso è lib64
install(TARGETS facial_capture facial_training facial_test DESTINATION /usr/local/bin)
install(FILES pam_facial.conf DESTINATION /etc/security)
