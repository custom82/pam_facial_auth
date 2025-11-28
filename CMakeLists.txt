cmake_minimum_required(VERSION 3.20)
project(pam_facial_auth LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Output dirs
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

include(GNUInstallDirs)

#
# ============================================================
#  TROVA OPENCV
# ============================================================
#
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

message(STATUS "OpenCV Version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")

#
# ============================================================
#  RILEVAMENTO CUDA DA OPENCV (FUNZIONA SU GENTOO)
# ============================================================
#

set(USE_CUDA OFF)
set(OPENCV_CUDA_LIB_FOUND OFF)

foreach(lib ${OpenCV_LIBS})
    if (lib MATCHES "cudaarithm" OR
        lib MATCHES "cudawarping" OR
        lib MATCHES "cudaimgproc" OR
        lib MATCHES "cudafeatures2d" OR
        lib MATCHES "cudafilters" OR
        lib MATCHES "cudaoptflow" OR
        lib MATCHES "cudacodec")
        set(OPENCV_CUDA_LIB_FOUND ON)
    endif()
endforeach()

if (OPENCV_CUDA_LIB_FOUND)
    message(STATUS "Rilevato OpenCV con moduli CUDA → attivo CUDA")
    set(USE_CUDA ON)
else()
    message(STATUS "OpenCV senza moduli CUDA → CPU-only")
endif()

#
# ============================================================
#  TOOLKIT CUDA (solo se USE_CUDA=ON)
# ============================================================
#

if (USE_CUDA)
    # Gentoo installa CUDA qui:
    set(CUDAToolkit_ROOT "/opt/cuda-13.0.2")

    message(STATUS "Forzo CUDAToolkit_ROOT=${CUDAToolkit_ROOT}")

    find_package(CUDAToolkit REQUIRED)
    include_directories(${CUDAToolkit_INCLUDE_DIRS})

    message(STATUS "CUDA found: ${CUDAToolkit_VERSION}")

    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler")
endif()

#
# ============================================================
#  PAM
# ============================================================
#

find_package(PkgConfig REQUIRED)
pkg_check_modules(PAM REQUIRED pam)
include_directories(${PAM_INCLUDE_DIRS})

include_directories(include)

#
# ============================================================
#  LIBRERIA PRINCIPALE
# ============================================================
#

add_library(facialauth SHARED
    src/libfacialauth.cpp
)

if (USE_CUDA)
    target_link_libraries(facialauth
        ${OpenCV_LIBS}
        CUDA::cudart
    )
else()
    target_link_libraries(facialauth
        ${OpenCV_LIBS}
    )
endif()

#
# ============================================================
#  MODULO PAM
# ============================================================
#

add_library(pam_facial_auth MODULE
    src/pam_facial_auth.cpp
)

set_target_properties(pam_facial_auth PROPERTIES PREFIX "")

target_link_libraries(pam_facial_auth
    facialauth
    ${PAM_LIBRARIES}
)

#
# ============================================================
#  BINARI CLI
# ============================================================
#

add_executable(facial_capture src/facial_capture.cpp)
target_link_libraries(facial_capture facialauth ${OpenCV_LIBS})

add_executable(facial_training src/facial_training.cpp)
target_link_libraries(facial_training facialauth ${OpenCV_LIBS})

add_executable(facial_test src/facial_test.cpp)
target_link_libraries(facial_test facialauth ${OpenCV_LIBS})

#
# ============================================================
#  INSTALLAZIONE
# ============================================================
#

# PAM module
install(TARGETS pam_facial_auth
    LIBRARY DESTINATION /lib64/security
)

# Main library
install(TARGETS facialauth
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# CLI tools
install(TARGETS facial_capture facial_training facial_test
    RUNTIME DESTINATION /usr/sbin
)

# Headers
install(DIRECTORY include/
    DESTINATION /usr/include
)

# Config file
if(EXISTS "${CMAKE_SOURCE_DIR}/etc/pam_facial.conf")
    install(FILES etc/pam_facial.conf
        DESTINATION /etc/security
        PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
    )
endif()

# Man pages
install(
    FILES man/facial_capture.1 man/facial_training.1 man/facial_test.1
    DESTINATION ${CMAKE_INSTALL_MANDIR}/man1
)

install(
    FILES man/pam_facial_auth.8
    DESTINATION ${CMAKE_INSTALL_MANDIR}/man8
)
