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
include_directories(${CMAKE_SOURCE_DIR}/src)  # Aggiungi la tua directory di sorgenti

# Aggiungi la libreria di utilità
add_library(Utils STATIC ${CMAKE_SOURCE_DIR}/src/Utils.cpp)

# Creazione del modulo PAM come libreria condivisa
add_library(pam_facial_auth MODULE
    ${CMAKE_SOURCE_DIR}/src/pam_facial_auth.cpp  # Assicurati che il file sia presente
    ${CMAKE_SOURCE_DIR}/src/FaceRecWrapper.cpp
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

# Impostare l'output della libreria
set_target_properties(pam_facial_auth PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# Compila il binario facial_test
add_executable(facial_test
    ${CMAKE_SOURCE_DIR}/src/facial_test.cpp  # Assicurati che il file sia presente
    ${CMAKE_SOURCE_DIR}/src/FaceRecWrapper.cpp
)

# Link con le librerie necessarie per facial_test
target_link_libraries(facial_test
    PRIVATE
    ${OpenCV_LIBS}
    Utils
)

# Compila il binario facial_training
add_executable(facial_training
    ${CMAKE_SOURCE_DIR}/src/facial_training.cpp  # Assicurati che il file sia presente
    ${CMAKE_SOURCE_DIR}/src/FaceRecWrapper.cpp
)

# Link con le librerie necessarie per facial_training
target_link_libraries(facial_training
    PRIVATE
    ${OpenCV_LIBS}
    Utils
)

# Impostare l'output dei binari
set_target_properties(facial_test facial_training pam_facial_auth PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Installazione dei file
install(TARGETS facial_test facial_training pam_facial_auth
    DESTINATION /usr/sbin
)

# Installazione della libreria
install(TARGETS pam_facial_auth
    DESTINATION /usr/lib64/security
)

# Installazione della libreria Utils
install(TARGETS Utils
    DESTINATION /usr/lib64
)

# Installazione dei file XML
install(FILES etc/haarcascade_frontalface_alt.xml etc/haarcascade_frontalface_default.xml
    DESTINATION /etc/pam_facial_auth
)
