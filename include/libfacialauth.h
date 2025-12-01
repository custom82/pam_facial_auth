#ifndef LIBFACIALAUTH_H
#define LIBFACIALAUTH_H

#include <string>
#include <map>
#include <vector>
#include <opencv2/core.hpp>

// Path di default per il file di configurazione PAM
#ifndef FACIALAUTH_DEFAULT_CONFIG
#define FACIALAUTH_DEFAULT_CONFIG "/etc/security/pam_facial.conf"
#endif


// Struttura di configurazione globale
struct FacialAuthConfig {
    // Percorsi / generali
    std::string basedir            = "/var/lib/pam_facial_auth";
    std::string device             = "/dev/video0";
    bool        fallback_device    = true;

    int  width      = 640;
    int  height     = 480;
    int  frames     = 20;
    int  sleep_ms   = 100;

    bool debug              = false;
    bool nogui              = true;
    std::string training_method = "auto"; // auto | lbph | eigen | fisher | sface
    bool force_overwrite    = false;
    bool ignore_failure     = false;
    bool save_failed_images = false;
    std::string image_format = "jpg";

    // Profili detector / recognizer
    std::string detector_profile   = "auto";       // auto | haar | yunet | yunet_int8 ...
    std::string recognizer_profile = "sface_fp32"; // sface_fp32 | sface_int8 | lbph | ...

    // Threshold per metodi "classici"
    double lbph_threshold    = 60.0;
    double eigen_threshold   = 5000.0;
    double fisher_threshold  = 500.0;
    int    eigen_components  = 80;
    int    fisher_components = 80;

    // Threshold per SFace
    double sface_threshold        = 0.5; // back-compat
    double sface_fp32_threshold   = 0.5;
    double sface_int8_threshold   = 0.5;

    // Backend DNN
    std::string dnn_backend = "cpu";  // cpu | cuda | cuda_fp16 | opencl
    std::string dnn_target  = "cpu";  // cpu | cuda | cuda_fp16 | opencl

    // Campi legacy / singoli
    std::string model_path;
    std::string haar_cascade_path;
    std::string yunet_model;
    std::string yunet_model_int8;
    std::string sface_model;
    std::string sface_model_int8;

    // Map dinamiche
    std::map<std::string, std::string> detector_models;   // es. "haar", "yunet_fp32", "yunet_int8"
    std::map<std::string, std::string> recognizer_models; // es. "sface_fp32", "sface_int8"
};

// Carica configurazione da file (es. /etc/security/pam_facial.conf)
bool fa_load_config(
    FacialAuthConfig &cfg,
    std::string &logbuf,
    const std::string &path
);

// Percorsi per immagini e modelli utente
std::string fa_user_image_dir(
    const FacialAuthConfig &cfg,
    const std::string &user
);

std::string fa_user_model_path(
    const FacialAuthConfig &cfg,
    const std::string &user
);

// API principali

// Cattura immagini di training
bool fa_capture_images(
    const std::string &user,
    const FacialAuthConfig &cfg,
    const std::string &format,
    std::string &log
);

// Allena modello per utente
bool fa_train_user(
    const std::string &user,
    const FacialAuthConfig &cfg,
    std::string &log
);

// Testa utente
bool fa_test_user(
    const std::string &user,
    const FacialAuthConfig &cfg,
    const std::string &modelPath,
    double &best_conf,
    int &best_label,
    std::string &log,
    double threshold_override = -1.0
);

// Utilità
bool fa_check_root(const char *tool_name);


#endif // LIBFACIALAUTH_H
