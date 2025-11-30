#ifndef LIBFACIALAUTH_H
#define LIBFACIALAUTH_H

#include <string>

#define FACIALAUTH_CONFIG_DEFAULT "/etc/security/pam_facial.conf"

/**
 * FacialAuth main configuration structure.
 *
 * It is filled by fa_load_config() using the config file,
 * but CLI tools (facial_capture, facial_training, facial_test)
 * may override values at runtime.
 *
 * The struct is intentionally generic; detector/recognizer profiles
 * and model paths are resolved at runtime according to the configuration.
 */
struct FacialAuthConfig {

    // =============================================================================
    // Base directory for image storage, user models, etc.
    // Example:
    //   basedir/images/<user>/*.jpg
    //   basedir/models/<user>.xml
    // =============================================================================
    std::string basedir;

    // =============================================================================
    // Camera parameters
    // =============================================================================
    std::string device;          // main /dev/videoX device
    bool fallback_device = true; // try /dev/video0..2 if main fails

    int width  = 640;            // capture width
    int height = 480;            // capture height
    int frames = 15;             // number of frames to capture/test
    int sleep_ms = 50;           // delay between frames

    // =============================================================================
    // Logging and runtime behaviour
    // =============================================================================
    bool debug = false;          // verbose stderr debugging
    bool nogui = false;          // GUI disabled (reserved)

    std::string model_path;      // optional override for model file
    std::string haar_cascade_path;
    std::string training_method = "lbph";
    std::string log_file;        // unused now (legacy, no syslog)

    bool force_overwrite = false;
    bool ignore_failure  = false;

    // =============================================================================
    // Classic recognizers (LBPH / Eigen / Fisher)
    // =============================================================================
    double lbph_threshold   = 80.0;
    double eigen_threshold  = 3000.0;
    double fisher_threshold = 500.0;

    int eigen_components  = 80;
    int fisher_components = 50;

    // =============================================================================
    // Detector (Haar / YuNet)
    //
    // detector_profile accepted values:
    //   "auto", "haar", "yunet_fp32", "yunet_int8"
    //
    // Yunet backend is internal logic; OpenCV DNN backend is controlled by
    // dnn_backend and dnn_target.
    // =============================================================================
    std::string detector_profile;

    // YuNet backend name (historical; kept for compatibility)
    std::string yunet_backend;

    // DNN execution backend/target:
    //   dnn_backend accepted values:
    //       auto, cpu, cuda, cuda_fp16, opencl
    //   dnn_target accepted values:
    //       auto, cpu, cuda, cuda_fp16, opencl
    std::string dnn_backend;
    std::string dnn_target;

    // Detector models (YuNet ONNX)
    std::string yunet_model;        // FP32 model
    std::string yunet_model_int8;   // INT8 model

    // =============================================================================
    // Recognizers (SFace DNN or classical)
    //
    // recognizer_profile accepted values:
    //   sface_fp32, sface_int8,
    //   lbph, eigen, fisher
    // =============================================================================
    std::string recognizer_profile = "sface";

    // DNN recognizer models (SFace ONNX)
    std::string sface_model;        // FP32
    std::string sface_model_int8;   // INT8
    double      sface_threshold = 0.5;

    // =============================================================================
    // Extra options
    // =============================================================================
    bool save_failed_images = false;

    // =============================================================================
    // Image format used by facial_capture (jpg/png)
    // =============================================================================
    std::string image_format = "jpg";
};


// =================================================================================
// Library API
// =================================================================================

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

// Utility: ensures the tool is executed as root
bool fa_check_root(const char *tool_name);

// CLI entrypoints (legacy wrappers)
int facial_capture_main(int argc, char *argv[]);
int facial_training_cli_main(int argc, char *argv[]);
int facial_test_cli_main(int argc, char *argv[]);

#endif // LIBFACIALAUTH_H
