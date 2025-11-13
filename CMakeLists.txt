cmake_minimum_required(VERSION 3.10)

project(pam_facial_auth CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# -------------------------------------------------------
# TROVA OPENCV E PAM
# -------------------------------------------------------
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(PAM REQUIRED pam)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${PAM_INCLUDE_DIRS}
)

# -------------------------------------------------------
# LIBRERIA CONDIVISA (libfacialauth.so)
# -------------------------------------------------------
add_library(facialauth SHARED
    src/libfacialauth.cpp
)

target_link_libraries(facialauth
    ${OpenCV_LIBS}
    ${PAM_LIBRARIES}
)

# -------------------------------------------------------
# EXECUTABLE: facial_capture
# Non include OpenCV, linka solo libfacialauth
# -------------------------------------------------------
add_executable(facial_capture
    src/facial_capture.cpp
)

target_include_directories(facial_capture PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(facial_capture
    facialauth
)

# -------------------------------------------------------
# EXECUTABLE: facial_training
# -------------------------------------------------------
add_executable(facial_training
    src/facial_training.cpp
)

target_include_directories(facial_training PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(facial_training
    facialauth
)

# -------------------------------------------------------
# EXECUTABLE: facial_test (se lo usi)
# -------------------------------------------------------
if(EXISTS "${PROJECT_SOURCE_DIR}/src/facial_test.cpp")
    add_executable(facial_test
        src/facial_test.cpp
    )

    target_include_directories(facial_test PRIVATE
        ${PROJECT_SOURCE_DIR}/include
    )

    target_link_libraries(facial_test
        facialauth
    )
endif()

# -------------------------------------------------------
# MODULO PAM (pam_facial_auth.so)
# -------------------------------------------------------
add_library(pam_facial_auth MODULE
    src/pam_facial_auth.cpp
)

target_include_directories(pam_facial_auth PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(pam_facial_auth
    facialauth
    ${PAM_LIBRARIES}
)

# Installazione modulo PAM
install(TARGETS pam_facial_auth
    LIBRARY DESTINATION /lib64/security
)

# Installazione libreria
install(TARGETS facialauth
    LIBRARY DESTINATION /usr/lib64
)

# Installazione binari
install(TARGETS facial_capture facial_training
    RUNTIME DESTINATION /usr/bin
)
