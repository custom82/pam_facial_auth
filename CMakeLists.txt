cmake_minimum_required(VERSION 3.10)
project(pam_facial_auth CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED core imgproc objdetect videoio highgui dnn face)

# PAM (path classico su Gentoo)
find_library(PAM_LIB pam REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include)

# =========================
# Libreria condivisa comune
# =========================
add_library(facialauth SHARED
    src/Utils.cpp
    src/FacialAuth.cpp
    src/FaceRecWrapper.cpp
)

target_link_libraries(facialauth
    PRIVATE
    ${OpenCV_LIBS}
)

target_include_directories(facialauth PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

set_target_properties(facialauth PROPERTIES
    OUTPUT_NAME facialauth
    SOVERSION 1
    VERSION 1.0.0
)

# =========================
# Modulo PAM
# =========================
add_library(pam_facial_auth MODULE
    src/pam_facial_auth.cpp
)

target_link_libraries(pam_facial_auth
    PRIVATE
    facialauth
    ${OpenCV_LIBS}
    ${PAM_LIB}
)

set_target_properties(pam_facial_auth PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# =========================
# Tool: facial_training
# =========================
add_executable(facial_training
    src/facial_training.cpp
)
target_link_libraries(facial_training PRIVATE facialauth ${OpenCV_LIBS})
set_target_properties(facial_training PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# =========================
# Tool: facial_test
# =========================
add_executable(facial_test
    src/facial_test.cpp
)
target_link_libraries(facial_test PRIVATE facialauth ${OpenCV_LIBS})
set_target_properties(facial_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# =========================
# Tool: facial_capture
# =========================
add_executable(facial_capture
    src/facial_capture.cpp
)
target_link_libraries(facial_capture PRIVATE facialauth ${OpenCV_LIBS})
set_target_properties(facial_capture PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# =========================
# Install (opzionale)
# =========================
install(TARGETS facialauth DESTINATION lib)
install(TARGETS pam_facial_auth DESTINATION lib64/security)
install(TARGETS facial_training facial_test facial_capture RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
