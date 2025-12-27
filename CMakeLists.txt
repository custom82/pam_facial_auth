cmake_minimum_required(VERSION 3.10)
project(pam_facial_auth)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc objdetect videoio face dnn)
find_package(PkgConfig REQUIRED)
pkg_check_modules(PAM REQUIRED pam)

include_directories(include ${OpenCV_INCLUDE_DIRS} ${PAM_INCLUDE_DIRS})

add_library(facialauth SHARED src/libfacialauth.cpp)
target_link_libraries(facialauth ${OpenCV_LIBS})

add_library(pam_facial_auth MODULE src/pam_facial_auth.cpp)
set_target_properties(pam_facial_auth PROPERTIES PREFIX "")
target_link_libraries(pam_facial_auth facialauth ${PAM_LIBRARIES})

set(TOOLS facial_capture facial_training facial_test)
foreach(TOOL ${TOOLS})
    add_executable(${TOOL} src/${TOOL}.cpp)
    target_link_libraries(${TOOL} facialauth ${OpenCV_LIBS})
endforeach()
