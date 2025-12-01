#ifndef LIBFACIALAUTH_H
#define LIBFACIALAUTH_H

#include <string>
#include <map>

#define FACIALAUTH_CONFIG_DEFAULT "/etc/security/pam_facial.conf"

/**
 * Main configuration structure for the FacialAuth library.
 * It is filled by fa_load_config() and can be overridden by CLI tools.
 */
struct FacialAuthConfig {
    // -------------------------------------------------------------------------
    // Base directory for images and models
    //   basedir/images/<user>/*.jpg
    //   basedir/models/<user>.xml
    // -------------------------------------------------------------------------
    std::string basedir;

    // -------------------------------------------------------------------------
    // Camera parameters
    // -------------------------------------------------------------------------
    std::string device;          // e.g. /dev/video0
    bool        fallback_device = true;

    int width    = 640;
    int height   = 480;
    int frames   = 15;
    int sleep_ms = 50;

    // -------------------------------------------------------------------------
    // Logging and runtime options
    // -------------------------------------------------------------------------
    bool debug = false;
    bool nogui = true;

    // Optional general model path (legacy / classic)
    std::string model_path;

    // Haar cascade path (legacy / classic detectors)
    std::string haar_cascade_path;

    // Training method (lbph / eigen / fisher / sface / auto)
    std::string training_method = "lbph";

    // Optional log file path (kept for compatibility, even if not used)
    std::string log_file;

    bool force_overwrite = false;
    bool ignore_failure  = false;

    // -------------------------------------------------------------------------
    // Classic recognizers (LBPH / Eigen / Fisher)
    // -------------------------------------------------------------------------
    double lbph_threshold   = 80.0;
    double eigen_threshold  = 3000.0;
    double fisher_threshold = 500.0;

    int eigen_components  = 80;
    int fisher_components = 50;

    // -------------------------------------------------------------------------
    // Detector and DNN backend
    // -------------------------------------------------------------------------
    // detector_profile:
    //   "auto" / "haar" / "yunet_fp32" / "yunet_int8" / others from config
    std::string detector_profile;

    // DNN backend/target for both YuNet and SFace:
    //   backend: "auto" / "cpu" / "cuda" / "cuda_fp16" / "opencl"
    //   target:  "auto" / "cpu" / "cuda" / "cuda_fp16" / "opencl"
    std::string dnn_backend;
    std::string dnn_target;

    // Explicit YuNet model paths (kept for compatibility with older config keys)
    std::string yunet_model;       // typically FP32
    std::string yunet_model_int8;  // INT8 variant

    // -------------------------------------------------------------------------
    // DNN recognizer (SFace)
    // -------------------------------------------------------------------------
    // recognizer_profile:
    //   "sface_fp32" / "sface_int8" / "lbph" / "eigen" / "fisher" / "auto"
    std::string recognizer_profile = "sface_fp32";

    // Direct SFace model paths (kept for compatibility with older config keys)
    std::string sface_model;       // FP32
    std::string sface_model_int8;  // INT8

    // Cosine similarity threshold for SFace
    double sface_threshold = 0.65;

    // -------------------------------------------------------------------------
    // Dynamic model maps from configuration
    // -------------------------------------------------------------------------
    // These maps are filled from keys prefixed with:
    //   detect_*      → detector_models
    //   recognize_*   → recognizer_models
    //
    // Example:
    //   detect_haar=/path/to/haar.xml
    //   detect_yunet_model_fp32=/path/to/yunet_fp32.onnx
    //   recognize_sface_model_fp32=/path/to/sface_fp32.onnx
    //
    // CLI tools can use these to print dynamic help and to resolve
    // available profiles without hardcoding them.
    std::map<std::string, std::string> detector_models;
    std::map<std::string, std::string> recognizer_models;

    // -------------------------------------------------------------------------
    // Misc options
    // -------------------------------------------------------------------------
    bool        save_failed_images = false;
    std::string image_format       = "jpg";
};

// ============================================================================
// Public API
// ============================================================================

/**
 * Load configuration from the given path into cfg.
 * Returns true on success, false on error.
 * logbuf will contain a human-readable log of what was parsed/overridden.
 */
bool fa_load_config(
    FacialAuthConfig &cfg,
    std::string &logbuf,
    const std::string &path = FACIALAUTH_CONFIG_DEFAULT
);

/**
 * Build the directory where user images are stored:
 *   <basedir>/images/<user>
 */
std::string fa_user_image_dir(
    const FacialAuthConfig &cfg,
    const std::string &user
);

/**
 * Build the path where the user model is stored:
 *   <basedir>/models/<user>.xml
 */
std::string fa_user_model_path(
    const FacialAuthConfig &cfg,
    const std::string &user
);

/**
 * Capture images from camera and store them in:
 *   <basedir>/images/<user>/img_XXXX.<format>
 */
bool fa_capture_images(
    const std::string &user,
    const FacialAuthConfig &cfg,
    const std::string &format,
    std::string &log
);

/**
 * Train the recognition model for a user.
 * For SFace, this computes an embedding and stores it in XML.
 * For classical methods, it trains LBPH/Eigen/Fisher models.
 */
bool fa_train_user(
    const std::string &user,
    const FacialAuthConfig &cfg,
    std::string &logbuf
);

/**
 * Test a user model against live camera frames.
 * Returns true on successful match (according to threshold).
 *
 * threshold_override:
 *   < 0 → use cfg.sface_threshold (or classic thresholds)
 *  >= 0 → use this value as effective threshold.
 */
bool fa_test_user(
    const std::string &user,
    const FacialAuthConfig &cfg,
    const std::string &modelPath,
    double &best_conf,
    int &best_label,
    std::string &logbuf,
    double threshold_override
);

/**
 * Simple helper to ensure the tool is running as root.
 * Returns true if uid == 0, false otherwise (and prints an error).
 */
bool fa_check_root(const char *tool_name);

// ============================================================================
// CLI entrypoints (wrappers around the library for binaries)
//
//  - facial_capture_main   → "facial_capture" tool
//  - facial_training_cli_main → "facial_training" tool
//  - facial_test_cli_main  → "facial_test" tool
// ============================================================================

int facial_capture_main(int argc, char *argv[]);
int facial_training_cli_main(int argc, char *argv[]);
int facial_test_cli_main(int argc, char *argv[]);

#endif // LIBFACIALAUTH_H
