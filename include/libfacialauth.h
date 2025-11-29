#ifndef LIBFACIALAUTH_H
#define LIBFACIALAUTH_H

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

// ==========================================================
// CONFIG
// ==========================================================

struct FacialAuthConfig {

    // Directory base di lavoro
    //  - default: /etc/pam_facial_auth (o simile)
    std::string basedir = "/etc/pam_facial_auth";

    // Dispositivo video
    std::string device = "/dev/video0";
    bool        fallback_device = true;   // prova /dev/video1 ecc. se fallisce

    // Risoluzione e cattura
    int  width    = 640;
    int  height   = 480;
    int  frames   = 10;       // numero di frame da catturare/testare
    int  sleep_ms = 100;      // pausa tra frame (ms)

    // Debug / GUI
    bool debug = false;
    bool nogui = true;

    // Modello "classico" (LBPH/EIGEN/FISHER)
    std::string model_path;         // override opzionale del path
    std::string haar_cascade_path;  // path assoluto cascata HAAR
    std::string training_method = "lbph"; // lbph/eigen/fisher/auto
    std::string log_file;           // file di log

    bool force_overwrite = false;   // sovrascrive modelli esistenti
    bool ignore_failure  = false;   // per CLI

    // Soglie per i metodi classici
    double lbph_threshold   = 60.0;
    double eigen_threshold  = 3500.0;
    double fisher_threshold = 500.0;

    int eigen_components  = 0;      // 0 = auto
    int fisher_components = 0;

    // -------------------------------
    // DNN / YuNet
    // -------------------------------

    // "haar" | "yunet" | "auto"
    std::string detector_profile = "auto";

    // backend yuNet/DNN: "cpu" | "cpu_int8"
    std::string yunet_backend = "cpu";

    // modelli YuNet
    std::string yunet_model;        // FP32
    std::string yunet_model_int8;   // INT8 per CPU

    // Alias generico per backend DNN (per compat)
    std::string dnn_backend;        // se valorizzato, può copiare in yunet_backend

    // -------------------------------
    // SFace (DNN face recognition)
    // -------------------------------
    // Profilo riconoscitore:
    // "lbph" | "eigen" | "fisher" | "sface" | "sface_int8"
    std::string recognizer_profile = "sface";

    // Modelli SFace ONNX
    std::string sface_model;        // modello FP32
    std::string sface_model_int8;   // modello INT8
    double      sface_threshold = 0.5; // soglia raccomandata ~0.5

    bool        save_failed_images = false;
};

#define FACIALAUTH_CONFIG_DEFAULT "/etc/security/pam_facial.conf"

// ==========================================================
// UTILS / API di alto livello
// ==========================================================

// Carica la configurazione da file.
// path vuoto -> FACIALAUTH_CONFIG_DEFAULT.
// logbuf (opzionale) accumula messaggi di log/errore.
bool fa_load_config(FacialAuthConfig &cfg,
                    std::string &logbuf,
                    const std::string &path = FACIALAUTH_CONFIG_DEFAULT);

// Directory immagini utente, es: basedir + "/images/<user>"
std::string fa_user_image_dir(const FacialAuthConfig &cfg,
                              const std::string &user);

// Path modello utente, es: basedir + "/models/<user>.xml"
std::string fa_user_model_path(const FacialAuthConfig &cfg,
                               const std::string &user);

// Cattura immagini dalla webcam e le salva in fa_user_image_dir(...) per training
bool fa_capture_images(const std::string &user,
                       const FacialAuthConfig &cfg,
                       const std::string &format,
                       std::string &logbuf);

// Esegue il training per l'utente usando le immagini in fa_user_image_dir
// - per lbph/eigen/fisher -> salva modello classico XML
// - per sface/sface_int8  -> salva file XML con embeddings DNN
bool fa_train_user(const std::string &user,
                   const FacialAuthConfig &cfg,
                   std::string &logbuf);

// Test/Autenticazione utente
//  - modelPath: se vuoto e uso classico -> fa_user_model_path(cfg,user)
//  - per SFace: se modelPath vuoto -> usa solo immagini; se è un modello
//    "sface" compatibile, carica embeddings da XML.
//  - best_conf: migliore conf/sim trovata
//  - best_label: label riconosciuta (solo per metodi classici)
//  - threshold_override: se >0, forza soglia
bool fa_test_user(const std::string &user,
                  const FacialAuthConfig &cfg,
                  const std::string &modelPath,
                  double &best_conf,
                  int &best_label,
                  std::string &logbuf,
                  double threshold_override = -1.0);

// Root check (per CLI)
bool fa_check_root(const char *tool_name);


// ==========================================================
// CLI WRAPPERS (definiti nel .cpp dei tool)
// ==========================================================

// facial_capture  -> cattura immagini
int facial_capture_main (int argc, char *argv[]);

// facial_training -> addestra modello (classico o SFace)
int facial_training_cli_main(int argc, char *argv[]);

// facial_test     -> test/auth locale, debug
int facial_test_cli_main    (int argc, char *argv[]);

#endif
