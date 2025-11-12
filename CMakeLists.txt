# Abilita C++20
set(CMAKE_CXX_STANDARD 20)

# Aggiungi le directory per l'inclusione dei file header
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    /usr/include/opencv4
)

# Aggiungi i file sorgenti per la libreria
set(LIBRARY_SOURCES
    src/pam_facial_auth.cpp
    src/FacialAuth.cpp
    src/FaceRecWrapper.cpp
    src/Utils.cpp
    src/facial_capture.cpp
    src/facial_training.cpp
)

# Crea una libreria condivisa chiamata libfacialauth.so per il codice comune (senza main())
add_library(facialauth SHARED ${LIBRARY_SOURCES})

# Linka la libreria PAM per il modulo PAM
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

# Aggiungi eseguibili separati per facial_test, facial_training, e facial_capture (con funzione main)
add_executable(facial_test src/facial_test.cpp)
target_link_libraries(facial_test PRIVATE facialauth)

add_executable(facial_training src/facial_training.cpp)
target_link_libraries(facial_training PRIVATE facialauth)

add_executable(facial_capture src/facial_capture.cpp)
target_link_libraries(facial_capture PRIVATE facialauth)

# Installazione dei file
install(TARGETS facialauth pam_facial_auth facial_test facial_training facial_capture
    LIBRARY DESTINATION /usr/lib
    RUNTIME DESTINATION /usr/bin
)

# Installazione delle man pages
install(FILES man/pam_facial_auth.1 DESTINATION /usr/share/man/man8)

# Installazione del file di configurazione
install(FILES ${CMAKE_SOURCE_DIR}/etc/pam_facial_auth/pam_facial.conf DESTINATION /etc/pam_facial_auth)

# Installazione della libreria .so con il nome corretto
install(TARGETS facialauth
    LIBRARY DESTINATION /usr/lib
    RENAME libfacialauth.so
)

# Installazione del modulo PAM
install(TARGETS pam_facial_auth
    LIBRARY DESTINATION /lib/security
)
