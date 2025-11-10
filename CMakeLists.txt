cmake_minimum_required(VERSION 3.10)

# Impostare il nome del progetto
project(pam_facial_auth)

# Aggiungi OpenCV e PAM come dipendenze
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(PAM REQUIRED pam)

# Aggiungi le directory di inclusione per PAM e OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PAM_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/src)  # Directory di sorgenti
include_directories(${CMAKE_SOURCE_DIR})      # Aggiungi la directory principale per facial_test e facial_training

# Aggiungi la libreria di utilità
add_library(Utils STATIC ${CMAKE_SOURCE_DIR}/src/Utils.cpp)

# Creazione del modulo PAM come libreria condivisa
add_library(pam_facial_auth MODULE
    src/FacialAuth.cpp
    src/FaceRecWrapper.cpp
)

# Imposta il nome del modulo come pam_facial_auth.so
set_target_properties(pam_facial_auth PROPERTIES
    PREFIX ""
    SUFFIX ".so"
)

# Link con le librerie necessarie per PAM e OpenCV
target_link_libraries(pam_facial_auth
    PRIVATE
    ${PAM_LIBRARIES}
    ${OpenCV_LIBS}
    Utils
)

# Compila il binario facial_test
add_executable(facial_test
    facial_test.cpp       # Non dentro la directory src
    src/FaceRecWrapper.cpp
)

# Link con le librerie necessarie per facial_test
target_link_libraries(facial_test
    PRIVATE
    ${OpenCV_LIBS}
    Utils
)

# Compila il binario facial_training
add_executable(facial_training
    facial_training.cpp   # Non dentro la directory src
    src/FaceRecWrapper.cpp
)

# Link con le librerie necessarie per facial_training
target_link_libraries(facial_training
    PRIVATE
    ${OpenCV_LIBS}
    Utils
)

# Impostare l'output dei binari sotto /usr/sbin
set_target_properties(facial_test facial_training pam_facial_auth PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY /usr/sbin
)

# Installare le man pages
install(FILES
    man/facial_test.1
    man/facial_training.1
    man/pam_facial_auth.1
    DESTINATION /usr/share/man/man1
)

# Installare i file XML
install(FILES
    etc/haarcascade_frontalface_alt.xml
    etc/haarcascade_frontalface_default.xml
    DESTINATION /etc/pam_facial_auth/
)

# Installare il modulo PAM
install(TARGETS pam_facial_auth
    DESTINATION /usr/lib64/security
)

# Installare i binari
install(TARGETS facial_test facial_training
    DESTINATION /usr/sbin
)

# Installare la libreria statica Utils.a
install(TARGETS Utils
    DESTINATION /usr/lib64
    ARCHIVE DESTINATION /usr/lib64
)

