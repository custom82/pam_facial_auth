#ifndef FACIALAUTH_LIB_H
#define FACIALAUTH_LIB_H

#include <string>
#include <map>

#define FACIALAUTH_CONFIG_DEFAULT "/etc/security/pam_facial.conf"

struct FacialAuthConfig {
    // Base
    std::string basedir;
    std::string device;
    bool        fallback_device = true;

    // Camera
    int width  = 640;
    int height = 480;
    int frames = 5;
    int sleep_ms = 200;

    // Runtime
    bool debug = false;
    bool nogui = false;

    // Generic
    std::string model_path;
    std::string training_method;
    std::string log_file;        // <<<<<<<<<< NECESSARIO

    bool force_overwrite = false;
    bool ignore_failure  = false;

    // Classic recognizers
    double lbph_threshold   = 80.0;
    double eigen_threshold  = 3000.0;
    double fisher_threshold = 500.0;

    int eigen_components  = 80;
    int fisher_components = 80;

    // Profiles
    std::string detector_profile;
    std::string recognizer_profile;

    // DNN backend/target
    std::string dnn_backend = "auto";
    std::string dnn_target  = "auto";

    // SFace
    double sface_threshold = 0.65;

    // File format
    bool        save_failed_images = false;
    std::string image_format = "jpg";

    // Detector paths (legacy)
    std::string haar_cascade_path;

    // NEW: maps for dynamic detectors / recognizers
    std::map<std::string,std::string> detector_models;    // <<<<<< REQUIRED
    std::map<std::string,std::string> recognizer_models;  // <<<<<< REQUIRED
};

// API =============================================================

bool fa_load_config(FacialAuthConfig &cfg,
                    std::string &logbuf,
                    const std::string &path = "");

std::string fa_user_image_dir(const FacialAuthConfig &cfg,
                              const std::string &user);

std::string fa_user_model_path(const FacialAuthConfig &cfg,
                               const std::string &user);

bool fa_capture_images(const std::string &user,
                       const FacialAuthConfig &cfg,
                       const std::string &format,
                       std::string &logbuf);

bool fa_train_user(const std::string &user,
                   const FacialAuthConfig &cfg,
                   std::string &logbuf);

bool fa_test_user(const std::string &user,
                  const FacialAuthConfig &cfg,
                  const std::string &modelPath,
                  double &best_conf,
                  int &best_label,
                  std::string &logbuf,
                  double threshold_override = -1.0);

bool fa_check_root(const char *tool_name);

#endif
