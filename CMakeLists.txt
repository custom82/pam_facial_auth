# Imposta il progetto
cmake_minimum_required(VERSION 3.10)
project(pam_facial_auth)

# Definisce il tipo di build
set(CMAKE_BUILD_TYPE RelWithDebInfo)

# Imposta la versione minima di OpenCV richiesta
find_package(OpenCV 4 REQUIRED)

# Imposta la libreria PAM
find_library(PAM_LIBRARIES NAMES pam)

# Includi la directory di OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})

# Aggiungi il file eseguibile per il test
add_executable(facial_test facial_test.cpp)

# Aggiungi il file eseguibile per il modulo PAM
add_library(pam_facial_auth SHARED src/FacialAuth.cpp src/FaceRecWrapper.cpp)

# Collega le librerie necessarie
target_link_libraries(facial_test
    ${OpenCV_LIBS}
    ${PAM_LIBRARIES}
)

target_link_libraries(pam_facial_auth
    ${OpenCV_LIBS}
    ${PAM_LIBRARIES}
)

# Aggiungi eventuali altre configurazioni di compilazione
set_target_properties(pam_facial_auth PROPERTIES PREFIX "")
