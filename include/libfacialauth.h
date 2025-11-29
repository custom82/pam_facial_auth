#ifndef LIBFACIALAUTH_H
#define LIBFACIALAUTH_H

#include <string>

// Path di default del file di configurazione PAM
// (come concordato: in /etc/security)
#define FACIALAUTH_CONFIG_DEFAULT "/etc/security/pam_facial.conf"

// ==========================================================
// Struttura di configurazione
// ==========================================================

struct FacialAuthConfig {
    // base dir per immagini e modelli
    std::string basedir = "/etc/pam_facial_auth";

    // dispositivo video
    std::string device  = "/dev/video0";

    // dimensioni e numero frame
    int  width     = 640;
    int  height    = 480;
    int  frames    = 30;
    int  sleep_ms  = 100;

    // flag generali
    bool debug           = false;
    bool nogui           = false;
    bool fallback_device = true;
    bool force_overwrite = false;

    // detector
    //  - "haar"
    //  - "yunet_cpu"
    //  - "yunet_cuda"
    std::string detector_profile = "haar";

    std::string haar_cascade_path; // es. /usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml
    std::string yunet_model;       // es. /usr/share/opencv4/dnn/models/face_detection_yunet_2023mar.onnx

    // SFace DNN
    // modello ONNX installato dall'ebuild:
    //  /usr/share/opencv4/dnn/models/face_recognition_sface_2021dec.onnx
    std::string sface_model;
    double      sface_threshold   = 0.5;

    // soglie modelli classici
    double lbph_threshold   = 80.0;
    double eigen_threshold  = 4000.0;
    double fisher_threshold = 400.0;

    // logging
    std::string log_file;
};

// ==========================================================
// API generiche
// ==========================================================

// Lettura file di configurazione stile key=value
bool read_kv_config(const std::string &path,
                    FacialAuthConfig  &cfg,
                    std::string       *logbuf);

// path helper
std::string fa_user_image_dir(const FacialAuthConfig &cfg,
                              const std::string      &user);

std::string fa_user_model_path(const FacialAuthConfig &cfg,
                               const std::string      &user);

// ==========================================================
// Funzioni di alto livello (usate anche da PAM)
// ==========================================================

// Cattura immagini per un utente
bool fa_capture(const std::string &user,
                const FacialAuthConfig &cfg_override,
                std::string &logbuf);

// Addestra un modello per l'utente
// method: "lbph" | "eigen" | "fisher" | "sface"
bool fa_train(const std::string &user,
              const FacialAuthConfig &cfg_override,
              const std::string &method,
              std::string &logbuf);

// Test di autenticazione (usato dal modulo PAM)
bool fa_test(const std::string &user,
             const FacialAuthConfig &cfg_override,
             double &confidence,
             std::string &logbuf);

// ==========================================================
// CLI (binari standalone)
// ==========================================================

int facial_capture_cli_main(int argc, char *argv[]);
int facial_training_cli_main(int argc, char *argv[]);
int facial_test_cli_main(int argc, char *argv[]);

// ==========================================================
// Helpers di manutenzione
// ==========================================================

bool fa_clean_images(const FacialAuthConfig &cfg, const std::string &user);
bool fa_clean_model (const FacialAuthConfig &cfg, const std::string &user);

void fa_list_images(const FacialAuthConfig &cfg, const std::string &user);

// Root check
bool fa_check_root(const char *tool_name);

#endif // LIBFACIALAUTH_H
