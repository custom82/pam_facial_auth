cmake_minimum_required(VERSION 3.10)
project(pam-facial-auth)

# Impostazione della versione del C++ a C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  # Disabilita estensioni del compilatore

# Ricerca di OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Usare pkg-config per trovare PAM
find_package(PkgConfig REQUIRED)
pkg_check_modules(PAM REQUIRED pam)

# Aggiungere le directory di include e libreria trovate con pkg-config
include_directories(${PAM_INCLUDE_DIRS} /usr/include/pam)  # Aggiungi manualmente il percorso degli header di PAM
link_directories(${PAM_LIBRARY_DIRS})

# Aggiungere i flag di compilazione e link con le librerie trovate da pkg-config
add_definitions(${PAM_CFLAGS})

# Creazione dell'eseguibile per l'addestramento del modello (rinominato)
add_executable(facial_training run_training.cpp src/FaceRecWrapper.cpp)
target_link_libraries(facial_training ${OpenCV_LIBS} ${PAM_LIBRARIES} -lpam_misc)  # Aggiungi -lpam_misc

# Creazione dell'eseguibile per il test del riconoscimento facciale (rinominato)
add_executable(facial_test run_test.cpp)
target_link_libraries(facial_test ${OpenCV_LIBS} ${PAM_LIBRARIES} -lpam_misc)  # Aggiungi -lpam_misc

# Creazione della libreria condivisa (per il modulo PAM)
option(COMPILE_SHARED_LIB "Compile PAM facial authentication module as a shared library" ON)

if(COMPILE_SHARED_LIB)
    add_library(pam_facial_auth SHARED src/FacialAuth.cpp src/FaceRecWrapper.cpp)
    target_link_libraries(pam_facial_auth ${OpenCV_LIBS} ${PAM_LIBRARIES} -lpam_misc)  # Aggiungi -lpam_misc
    set_target_properties(pam_facial_auth PROPERTIES PREFIX "")  # Il prefisso PAM è rimosso per essere solo "pam_facial_auth.so"
    install(TARGETS pam_facial_auth LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()  # Assicurati che la parentesi chiusa del blocco if sia presente qui

# Installazione degli eseguibili (opzionale)
install(TARGETS facial_training facial_test DESTINATION ${CMAKE_INSTALL_BINDIR})

# Definizione del percorso di installazione
set(CMAKE_INSTALL_PREFIX /usr/local)

# Aggiunta di eventuali file di configurazione PAM se necessario
install(FILES src/pam-facial-auth.conf DESTINATION ${CMAKE_INSTALL_SYSCONFDIR}/pam.d)

# Impostazioni per la libreria PAM
install(CODE "execute_process(COMMAND ldconfig)")
