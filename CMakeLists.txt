# Imposta la versione minima di CMake
cmake_minimum_required(VERSION 3.10)

# Definisci il nome del progetto
project(pam_facial_auth)

# Abilita C++20
set(CMAKE_CXX_STANDARD 20)

# Aggiungi le directory per l'inclusione dei file header
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    /usr/include/opencv4
)

# Aggiungi i file sorgenti
set(SOURCES
    src/pam_facial_auth.cpp
    src/FacialAuth.cpp
    src/FaceRecWrapper.cpp
    src/Utils.cpp
    src/facial_capture.cpp
    src/facial_test.cpp
    src/facial_training.cpp
)

# Crea una libreria condivisa per il codice comune
add_library(facialauth SHARED ${SOURCES})

# Aggiungi la libreria PAM come dipendenza di link
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
    /usr/lib64/libpam.so  # Aggiungi la libreria PAM per il modulo PAM
)

# Crea il modulo PAM come libreria condivisa
add_library(pam_facial_auth MODULE src/pam_facial_auth.cpp)

# Assicurati che il modulo PAM venga linkato correttamente con le librerie necessarie
target_link_libraries(pam_facial_auth
    PRIVATE
    facialauth  # Collega la libreria faccialauth per utilizzare le funzioni comuni
    /usr/lib64/libpam.so  # Collega la libreria PAM per il modulo PAM
)

# Imposta il percorso di output della libreria
set_target_properties(facialauth PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# Definisci il percorso di output del modulo PAM
set_target_properties(pam_facial_auth PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# Aggiungi un eseguibile per il test
add_executable(facial_test src/facial_test.cpp)

# Linka l'eseguibile al modulo e alle librerie necessarie
target_link_libraries(facial_test
    PRIVATE
    facialauth
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
)

# Aggiungi un target per il training
add_executable(facial_training src/facial_training.cpp)

# Linka l'eseguibile di training
target_link_libraries(facial_training
    PRIVATE
    facialauth
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
)

# Aggiungi un target per il facial capture
add_executable(facial_capture src/facial_capture.cpp)

# Linka l'eseguibile di capture
target_link_libraries(facial_capture
    PRIVATE
    facialauth
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
)

# Installazione dei file
install(TARGETS facialauth pam_facial_auth facial_test facial_training facial_capture
    LIBRARY DESTINATION /usr/lib
    RUNTIME DESTINATION /usr/bin
)

# Installazione delle man pages
install(FILES man/pam_facial_auth.1 DESTINATION /usr/share/man/man8)

# Installazione del file di configurazione
install(FILES ${CMAKE_SOURCE_DIR}/etc/pam_facial_auth/pam_facial.conf DESTINATION /etc/pam_facial_auth)

# Installazione della libreria .so
install(TARGETS facialauth
    LIBRARY DESTINATION /usr/lib
)

# Installazione del modulo PAM
install(TARGETS pam_facial_auth
    LIBRARY DESTINATION /lib/security
)
