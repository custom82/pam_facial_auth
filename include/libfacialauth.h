#ifndef LIBFACIALAUTH_H
#define LIBFACIALAUTH_H

#include <string>

#define FACIALAUTH_CONFIG_DEFAULT "/etc/security/pam_facial.conf"

/**
 * Main configuration structure for FacialAuth.
 * Filled by fa_load_config() and can be overridden by CLI tools.
 */
struct FacialAuthConfig {

    // -----------------------------------------------------------------------------
    // Base directory for images, models, etc.
    // -----------------------------------------------------------------------------
    std::string basedir;

    // -----------------------------------------------------------------------------
    // Camera parameters
    // -----------------------------------------------------------------------------
    std::string device;
    bool        fallback_device = true;

    int width    = 640;
    int height   = 480;
    int frames   = 15;
    int sleep_ms = 50;

    // -----------------------------------------------------------------------------
    // Runtime / logging
    // -----------------------------------------------------------------------------
    bool debug = false;
    bool nogui = false;

    std::string model_path;
    std::string haar_cascade_path;
    std::string training_method = "lbph";
    std::string log_file; // currently unused (kept for compatibility)

    bool force_overwrite = false;
    bool ignore_failure  = false;

    // -----------------------------------------------------------------------------
    // Classic models (LBPH / Eigen / Fisher)
    // -----------------------------------------------------------------------------
    double lbph_threshold   = 80.0;
    double eigen_threshold  = 3000.0;
    double fisher_threshold = 500.0;

    int eigen_components  = 80;
    int fisher_components = 50;

    // -----------------------------------------------------------------------------
    // Detector (HAAR / YuNet)
    // -----------------------------------------------------------------------------
    std::string detector_profile;
    std::string yunet_backend;
    std::string dnn_backend;
    std::string dnn_target;

    std::string yunet_model;
    std::string yunet_model_int8;

    // -----------------------------------------------------------------------------
    // DNN recognizer (SFace)
    // -----------------------------------------------------------------------------
    std::string recognizer_profile = "sface_fp32";

    std::string sface_model;
    std::string sface_model_int8;
    double      sface_threshold = 0.5;

    // -----------------------------------------------------------------------------
    // Extra options
    // -----------------------------------------------------------------------------
    bool save_failed_images = false;

    // -----------------------------------------------------------------------------
    // Image format used by facial_capture
    // -----------------------------------------------------------------------------
    std::string image_format = "jpg";
};

// =============================================================================
// Library API
// =============================================================================

bool fa_load_config(
    FacialAuthConfig &cfg,
    std::string &logbuf,
    const std::string &path
);

std::string fa_user_image_dir(
    const FacialAuthConfig &cfg,
    const std::string &user
);

std::string fa_user_model_path(
    const FacialAuthConfig &cfg,
    const std::string &user
);

bool fa_capture_images(
    const std::string &user,
    const FacialAuthConfig &cfg,
    const std::string &format,
    std::string &log
);

bool fa_train_user(
    const std::string &user,
    const FacialAuthConfig &cfg,
    std::string &logbuf
);

bool fa_test_user(
    const std::string &user,
    const FacialAuthConfig &cfg,
    const std::string &modelPath,
    double &best_conf,
    int &best_label,
    std::string &logbuf,
    double threshold_override
);

// Utilities
bool fa_check_root(const char *tool_name);

// CLI entrypoints (used by small wrapper binaries)
int facial_capture_main(int argc, char *argv[]);
int facial_training_cli_main(int argc, char *argv[]);
int facial_test_cli_main(int argc, char *argv[]);

// Optional: logging helpers (stderr-only)
void log_debug(const FacialAuthConfig &cfg, const char *fmt, ...);
void log_info (const FacialAuthConfig &cfg, const char *fmt, ...);
void log_error(const FacialAuthConfig &cfg, const char *fmt, ...);

#endif // LIBFACIALAUTH_H
