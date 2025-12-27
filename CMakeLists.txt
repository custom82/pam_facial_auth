cmake_minimum_required(VERSION 3.10)
project(pam_facial_auth)

# Standard C++17 necessario per std::filesystem
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Ricerca dipendenze
# Nota: includiamo 'dnn' per YuNet/SFace e 'face' per i plugin LBPH/Eigen/Fisher
find_package(OpenCV REQUIRED COMPONENTS
    core
    highgui
    imgproc
    objdetect
    videoio
    face
    dnn
)

find_package(PkgConfig REQUIRED)
pkg_check_modules(PAM REQUIRED pam)

# 2. Header
include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
    ${PAM_INCLUDE_DIRS}
)

# 3. LIBRERIA CORE (Contiene i plugin e la logica YuNet)
# Compilando questo file, compiliamo tutto il motore del sistema
add_library(facialauth SHARED src/libfacialauth.cpp)
target_link_libraries(facialauth ${OpenCV_LIBS})

# 4. MODULO PAM (Il plugin per /lib64/security/)
add_library(pam_facial_auth MODULE src/pam_facial_auth.cpp)
set_target_properties(pam_facial_auth PROPERTIES PREFIX "")
# Colleghiamo il modulo PAM alla nostra libreria core
target_link_libraries(pam_facial_auth facialauth ${PAM_LIBRARIES})

# 5. TOOL CLI (Wrapper leggeri)
set(TOOLS facial_capture facial_training facial_test)

foreach(TOOL ${TOOLS})
    add_executable(${TOOL} src/${TOOL}.cpp)
    # Colleghiamo i tool alla nostra lib e a OpenCV
    target_link_libraries(${TOOL} facialauth ${OpenCV_LIBS})

    # IMPORTANTE: Permette ai tool di trovare libfacialauth.so nella cartella di build
    # Durante l'installazione, l'ebuild di Gentoo gestirà il path finale in /usr/lib64
    set_target_properties(${TOOL} PROPERTIES SKIP_BUILD_RPATH FALSE)
    set_target_properties(${TOOL} PROPERTIES BUILD_WITH_INSTALL_RPATH FALSE)
endforeach()
