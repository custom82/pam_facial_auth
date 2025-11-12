cmake_minimum_required(VERSION 3.10)

project(pam_facial_auth)

set(CMAKE_CXX_STANDARD 20)

# Directory di inclusione
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(/usr/include/opencv4)

# Aggiungi la libreria senza gli eseguibili (senza main)
set(LIBRARY_SOURCES
    src/pam_facial_auth.cpp
    src/FacialAuth.cpp
    src/FaceRecWrapper.cpp
    src/Utils.cpp
    src/facial_capture.cpp  # Non include main()
    src/facial_training.cpp  # Non include main()
)

# Creazione della libreria condivisa
add_library(facialauth SHARED ${LIBRARY_SOURCES})

# Collegamento delle librerie necessarie
target_link_libraries(facialauth
    PRIVATE
    /usr/lib64/libopencv_highgui.so
    /usr/lib64/libopencv_face.so
    /usr/lib64/libopencv_videoio.so
    /usr/lib64/libopencv_imgcodecs.so
    /usr/lib64/libopencv_objdetect.so
    /usr/lib64/libopencv_dnn.so
    /usr/lib64/libopencv_calib3d.so
    /usr/lib64/libopencv_features2d.so
    /usr/lib64/libopencv_flann.so
    /usr/lib64/libopencv_photo.so
    /usr/lib64/libopencv_imgproc.so
    /usr/lib64/libopencv_core.so
    /usr/lib64/libpam.so
)

# Aggiungi gli eseguibili
add_executable(facial_test src/facial_test.cpp)
target_link_libraries(facial_test PRIVATE facialauth)

add_executable(facial_training src/facial_training.cpp)  # Con main()
target_link_libraries(facial_training PRIVATE facialauth)

add_executable(facial_capture src/facial_capture.cpp)  # Con main()
target_link_libraries(facial_capture PRIVATE facialauth)

# Aggiungi il modulo PAM
add_library(pam_facial_auth MODULE src/pam_facial_auth.cpp)
target_link_libraries(pam_facial_auth PRIVATE facialauth)

# Installazione dei binari
install(TARGETS facialauth facial_test facial_training facial_capture
    RUNTIME DESTINATION /usr/bin
    LIBRARY DESTINATION /usr/lib
)

# Installazione delle man pages
install(FILES man/pam_facial_auth.1 DESTINATION /usr/share/man/man8)

# Installazione del file di configurazione
install(FILES ${CMAKE_SOURCE_DIR}/etc/pam_facial_auth/pam_facial.conf DESTINATION /etc/pam_facial_auth)

# Installazione della libreria .so
install(TARGETS facialauth
    LIBRARY DESTINATION /usr/lib64
    RENAME libfacialauth.so
)

# Installazione del modulo PAM
install(TARGETS pam_facial_auth
    LIBRARY DESTINATION /usr/lib64/security
)
