#ifndef LIBFACIALAUTH_H
#define LIBFACIALAUTH_H

#include <string>
#include <vector>

#define FACIALAUTH_CONFIG_DEFAULT "/etc/security/pam_facial.conf"

/**
 * Main configuration structure for FacialAuth.
 * Filled via fa_load_config() and optionally overridden by CLI tools.
 */
struct FacialAuthConfig {

    // -------------------------------------------------------------------------
    // Base directory for images, models, etc.
    // -------------------------------------------------------------------------
    std::string basedir;

    // -------------------------------------------------------------------------
    // Camera parameters
    // -------------------------------------------------------------------------
    std::string device;
    bool        fallback_device = true;

    int width    = 640;
    int height   = 480;
    int frames   = 15;
    int sleep_ms = 50;

    // -------------------------------------------------------------------------
    // Logging and runtime options
    // -------------------------------------------------------------------------
    bool debug = false;
    bool nogui = false;

    std::string model_path;
    std::string haar_cascade_path;
    std::string training_method = "lbph";

    bool force_overwrite = false;
    bool ignore_failure  = false;

    // -------------------------------------------------------------------------
    // Classic models (LBPH / Eigen / Fisher)
    // -------------------------------------------------------------------------
    double lbph_threshold   = 80.0;
    double eigen_threshold  = 3000.0;
    double fisher_threshold = 500.0;

    int eigen_components  = 80;
    int fisher_components = 50;

    // -------------------------------------------------------------------------
    // Detector (Haar / YuNet)
    // -------------------------------------------------------------------------
    std::string detector_profile;      // auto | haar | yunet_fp32 | yunet_int8

    // For YuNet backend selection (CPU/CUDA)
    std::string yunet_backend;

    // DNN backend/target for SFace and YuNet
    std::string dnn_backend;           // auto | cpu | cuda | cuda_fp16 | opencl
    std::string dnn_target;            // auto | cpu | cuda | cuda_fp16 | opencl

    // YuNet models
    std::string yunet_model_fp32;
    std::string yunet_model_int8;

    // -------------------------------------------------------------------------
    // DNN recognizer (SFace)
    // -------------------------------------------------------------------------
    std::string recognizer_profile = "sface_fp32"; // sface_fp32 | sface_int8 | lbph | eigen | fisher

    std::string sface_model_fp32;
    std::string sface_model_int8;
    double      sface_threshold = 0.5;

    // -------------------------------------------------------------------------
    // Extra options
    // -------------------------------------------------------------------------
    bool save_failed_images = false;

    // -------------------------------------------------------------------------
    // Image format for facial_capture
    // -------------------------------------------------------------------------
    std::string image_format = "jpg";
};


// ============================================================================
// Public API
// ============================================================================

bool fa_load_config(
    FacialAuthConfig &cfg,
    std::string &logbuf,
    const std::string &path
);

// Paths for images and models
std::string fa_user_image_dir(
    const FacialAuthConfig &cfg,
    const std::string &user
);

std::string fa_user_model_path(
    const FacialAuthConfig &cfg,
    const std::string &user
);

// Capture / Train / Test
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

// Root check helper (for PAM-related tools)
bool fa_check_root(const char *tool_name);


// ============================================================================
// Video device enumeration (v4l2-ctl like)
// ============================================================================

struct FaVideoDeviceInfo {
    std::string dev_node;       // /dev/video0

    std::string card;           // From VIDIOC_QUERYCAP
    std::string driver;         // From VIDIOC_QUERYCAP
    std::string bus_info;       // From VIDIOC_QUERYCAP

    // From sysfs (USB/PCI if available)
    std::string manufacturer;
    std::string product;

    // USB IDs (if USB)
    std::string usb_vendor_id;  // hex, e.g. "04f2"
    std::string usb_product_id; // hex, e.g. "b79f"

    // PCI IDs (if PCI capture card)
    std::string pci_vendor_id;  // hex
    std::string pci_device_id;  // hex
};

/**
 * Enumerate all /dev/video* devices that support video capture.
 *
 * Returns true on success. Warnings and non-fatal problems are appended to logbuf.
 */
bool fa_list_video_devices(
    std::vector<FaVideoDeviceInfo> &devices,
    std::string &logbuf
);

/**
 * Enumerate "tested" resolutions for a given /dev/videoX node.
 * The function uses V4L2 and tries a fixed set of common resolutions.
 */
bool fa_list_device_resolutions(
    const std::string &dev_node,
    std::vector<std::pair<int,int>> &resolutions,
    std::string &logbuf
);

#endif // LIBFACIALAUTH_H
