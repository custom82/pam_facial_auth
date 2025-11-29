#ifndef FACIALAUTH_H
#define FACIALAUTH_H

#include <string>
#include <vector>
#include <opencv2/core.hpp>

#define FACIALAUTH_CONFIG_DEFAULT "/etc/security/pam_facial.conf"

struct FacialAuthConfig {
    // base working directory: usually /etc/pam_facial_auth
    std::string basedir = "/etc/pam_facial_auth";

    // camera settings
    std::string device = "/dev/video0";
    bool fallback_device = true;
    int width = 1280;
    int height = 720;
    int capture_count = 50;
    int capture_delay = 50; // ms

    // detector / recognizer
    std::string detector_profile = "yunet"; // auto|yunet|yunet_int8|haar|none
    std::string recognizer = "auto";        // auto|eigen|fisher|lbph|sface

    // DNN backend / target (for YuNet + future SFace)
    std::string dnn_backend = "cpu";        // cpu|cuda|auto

    // models
    std::string haar_cascade_path;
    std::string yunet_model;
    std::string yunet_model_int8;
    std::string sface_model;
    std::string sface_model_int8;

    // thresholds
    double eigen_threshold = 350.0;
    double fisher_threshold = 150.0;
    double lbph_threshold = 80.0;
    double sface_threshold = 0.40;

    // logging
    bool debug = false;
    bool save_failed_images = false;
    std::string log_file = "/var/log/pam_facial_auth.log";

    // behaviour
    bool force_overwrite = false;
};

// generic utils
bool fa_file_exists(const std::string &path);
bool fa_ensure_dir(const std::string &path);
void fa_msleep(int ms);

// config
bool fa_read_config(const std::string &path, FacialAuthConfig &cfg, std::string &err);

// high level paths
std::string fa_user_image_dir(const FacialAuthConfig &cfg, const std::string &user);
std::string fa_user_model_path(const FacialAuthConfig &cfg, const std::string &user);

// library API (used both by CLI tools and pam_facial_auth)
bool fa_capture_images(const FacialAuthConfig &cfg,
                       const std::string &user,
                       int max_images,
                       std::string &err);

bool fa_train_user(const FacialAuthConfig &cfg,
                   const std::string &user,
                   const std::string &method,
                   std::string &err);

bool fa_test_user(const FacialAuthConfig &cfg,
                  const std::string &user,
                  double &best_conf,
                  std::string &used_method,
                  std::string &err);

// CLI entry points (so we can keep main() in tiny wrappers)
int facial_capture_cli_main(int argc, char **argv);
int facial_training_cli_main(int argc, char **argv);
int facial_test_cli_main(int argc, char **argv);

#endif // FACIALAUTH_H
