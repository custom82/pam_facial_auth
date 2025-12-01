cmake_minimum_required(VERSION 3.16)
project(pam_facial_auth CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ========================================================
#  RPATH FIX — importantissimo su Gentoo + CUDA
# ========================================================
set(CMAKE_SKIP_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

# Directory CUDA (se abilitata)
set(CUDA_LIBDIR "/opt/cuda-13.0.2/targets/x86_64-linux/lib")
set(CUDA_INCDIR "/opt/cuda-13.0.2/include")

# ========================================================
#  Opzioni configurazione
# ========================================================
option(ENABLE_CUDA   "Enable CUDA backend"   OFF)
option(ENABLE_OPENCL "Enable OpenCL backend" OFF)

# ========================================================
#  OpenCV
# ========================================================
find_package(OpenCV REQUIRED
    PATHS /usr
    COMPONENTS core imgproc videoio imgcodecs dnn objdetect face
)

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})

# ========================================================
#  CUDA (se abilitata)
# ========================================================
if(ENABLE_CUDA)
    message(STATUS "CUDA ENABLED")
    include_directories(${CUDA_INCDIR})
    link_directories(${CUDA_LIBDIR})
endif()

# ========================================================
#  OpenCL (se abilitata)
# ========================================================
if(ENABLE_OPENCL)
    message(STATUS "OPENCL ENABLED")
    find_library(OPENCL_LIBRARY OpenCL REQUIRED)
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)

# ========================================================
#  Libreria condivisa libfacialauth.so
# ========================================================
add_library(facialauth SHARED
    src/libfacialauth.cpp
)

target_link_libraries(facialauth
    ${OpenCV_LIBS}
)

if(ENABLE_CUDA)
    target_link_libraries(facialauth
        ${CUDA_LIBDIR}/libcudart_static.a
        nppial nppitc nppig nppist nppidei nppicc nppif nppim nppc
        cublas culibos cublasLt cufft
    )
endif()

if(ENABLE_OPENCL)
    target_link_libraries(facialauth ${OPENCL_LIBRARY})
endif()

# Imposta un RPATH pulito
set_target_properties(facialauth PROPERTIES
    INSTALL_RPATH "${CUDA_LIBDIR}"
)

# ========================================================
#  PAM module pam_facial_auth.so
# ========================================================
add_library(pam_facial_auth SHARED
    src/pam_facial_auth.cpp
)

target_link_libraries(pam_facial_auth
    facialauth
    pam
    ${OpenCV_LIBS}
)

if(ENABLE_CUDA)
    target_link_libraries(pam_facial_auth
        ${CUDA_LIBDIR}/libcudart_static.a
        nppial nppitc nppig nppist nppidei nppicc nppif nppim nppc
        cublas culibos cublasLt cufft
    )
endif()

if(ENABLE_OPENCL)
    target_link_libraries(pam_facial_auth ${OPENCL_LIBRARY})
endif()

set_target_properties(pam_facial_auth PROPERTIES
    PREFIX ""                     # gera pam_facial_auth.so (senza lib)
    INSTALL_RPATH "${CUDA_LIBDIR}"
)

# ========================================================
#  Eseguibili
# ========================================================
add_executable(facial_capture   src/facial_capture.cpp)
add_executable(facial_training  src/facial_training.cpp)
add_executable(facial_test      src/facial_test.cpp)

foreach(bin facial_capture facial_training facial_test)
    target_link_libraries(${bin}
        facialauth
        ${OpenCV_LIBS}
    )

    if(ENABLE_CUDA)
        target_link_libraries(${bin}
            ${CUDA_LIBDIR}/libcudart_static.a
            nppial nppitc nppig nppist nppidei nppicc nppif nppim nppc
            cublas culibos cublasLt cufft
        )
    endif()

    if(ENABLE_OPENCL)
        target_link_libraries(${bin} ${OPENCL_LIBRARY})
    endif()

    set_target_properties(${bin} PROPERTIES
        INSTALL_RPATH "${CUDA_LIBDIR}"
    )
endforeach()

# ========================================================
#  Installazione
# ========================================================
install(TARGETS facialauth pam_facial_auth
    LIBRARY DESTINATION lib64/security
)

install(TARGETS facial_capture facial_training facial_test
    RUNTIME DESTINATION sbin
)

install(FILES etc/pam_facial.conf DESTINATION /etc/security)
install(FILES README.md LICENSE DESTINATION /usr/share/doc/pam_facial_auth)
