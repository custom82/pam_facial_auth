# Impostare la versione minima di CMake
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
include_directories(${CMAKE_SOURCE_DIR}/src)  # Aggiungi la directory di sorgenti
include_directories(${CMAKE_SOURCE_DIR})      # Aggiungi la directory principale per facial_test e facial_training

# Aggiungi la libreria di utilità
add_library(Utils STATIC ${CMAKE_SOURCE_DIR}/src/Utils.cpp)

# Creazione del modulo PAM come libreria condivisa
add_library(pam_facial_auth MODULE
    src/pam_facial_auth.c        # Include il file del modulo PAM
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

# Impostare l'output della libreria
set_target_properties(pam_facial_auth PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# Compilare il binario facial_test
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

# Compilare il binario facial_training
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

# Impostare l'output dei binari
set_target_properties(facial_test facial_training pam_facial_auth PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Installazione del modulo PAM, dei binari e delle man pages
install(TARGETS pam_facial_auth
    LIBRARY DESTINATION /usr/lib64/security
)

install(TARGETS facial_test facial_training
    RUNTIME DESTINATION /usr/sbin
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/man/
    DESTINATION /usr/share/man
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/etc/
    DESTINATION /etc/pam_facial_auth
)

# Installazione delle man pages
install(FILES ${CMAKE_SOURCE_DIR}/man/facial_test.1
    DESTINATION /usr/share/man/man1
)

install(FILES ${CMAKE_SOURCE_DIR}/man/facial_training.1
    DESTINATION /usr/share/man/man1
)

install(FILES ${CMAKE_SOURCE_DIR}/man/pam_facial_auth.8
    DESTINATION /usr/share/man/man8
)

# Aggiungere l'installazione della libreria di utilità
install(TARGETS Utils
    ARCHIVE DESTINATION /usr/lib64
)
