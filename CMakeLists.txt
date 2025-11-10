# Aggiungi OpenCV e PAM come dipendenze
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(PAM REQUIRED libpam)

# Aggiungi le directory di inclusione per PAM e OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PAM_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/src)  # Directory di sorgenti

# Aggiungi la libreria di utilità
add_library(Utils STATIC ${CMAKE_SOURCE_DIR}/src/Utils.cpp)

# Compila il modulo PAM
add_library(pam_facial_auth MODULE
    src/FacialAuth.cpp
    src/FaceRecWrapper.cpp
    src/Utils.cpp  # Aggiungi Utils.cpp qui se necessario
)

# Link con le librerie necessarie
target_link_libraries(pam_facial_auth
    PRIVATE
    ${PAM_LIBRARIES}
    ${OpenCV_LIBS}
    Utils
)

# Compila il binario facial_auth
add_executable(facial_auth
    facial_auth.cpp        # Percorso corretto, fuori dalla directory src
    src/FaceRecWrapper.cpp # Ancora nella directory src
)

# Link con le librerie necessarie per facial_auth
target_link_libraries(facial_auth
    PRIVATE
    ${OpenCV_LIBS}
    Utils
)

# Compila il binario facial_training
add_executable(facial_training
    facial_training.cpp    # Percorso corretto, fuori dalla directory src
    src/FaceRecWrapper.cpp # Ancora nella directory src
)

# Link con le librerie necessarie per facial_training
target_link_libraries(facial_training
    PRIVATE
    ${OpenCV_LIBS}
    Utils
)

# Impostare l'output dei binari (opzionale)
set_target_properties(facial_auth facial_training pam_facial_auth PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Impostare l'output dei binari (opzionale)
set_target_properties(facial_auth facial_training pam_facial_auth PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
