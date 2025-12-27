cmake_minimum_required(VERSION 3.10)
project(pam_facial_auth)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Cerca OpenCV con tutti i moduli necessari per la lib e i plugin
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc objdetect videoio face dnn)

# Cerca le librerie PAM tramite PkgConfig
find_package(PkgConfig REQUIRED)
pkg_check_modules(PAM REQUIRED pam)

# Includi le directory degli header
include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
    ${PAM_INCLUDE_DIRS}
)

# 1. LIBRERIA CORE (libfacialauth.so)
# Questo file contiene la logica dei plugin, YuNet e le API fa_*
add_library(facialauth SHARED src/libfacialauth.cpp)
target_link_libraries(facialauth ${OpenCV_LIBS})

# 2. MODULO PAM (pam_facial_auth.so)
# Deve essere un MODULE per essere caricato dinamicamente da PAM
add_library(pam_facial_auth MODULE src/pam_facial_auth.cpp)
set_target_properties(pam_facial_auth PROPERTIES PREFIX "")
target_link_libraries(pam_facial_auth facialauth ${PAM_LIBRARIES})

# 3. I TRE TOOL WRAPPER (CLI)
set(TOOLS facial_capture facial_training facial_test)

foreach(TOOL ${TOOLS})
    add_executable(${TOOL} src/${TOOL}.cpp)
    # Colleghiamo alla libreria core. I tool diventano wrapper leggeri.
    target_link_libraries(${TOOL} facialauth ${OpenCV_LIBS})
endforeach()
