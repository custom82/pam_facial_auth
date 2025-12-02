#pragma once

#include <string>
#include <vector>
#include <map>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/face.hpp>

// ==========================================================
// Default configuration file path
// ==========================================================
#ifndef FACIALAUTH_DEFAULT_CONFIG
#define FACIALAUTH_DEFAULT_CONFIG "/etc/security/pam_facial.conf"
#endif

// ==========================================================
// FacialAuthConfig: runtime configuration loaded from file
// ==========================================================
struct FacialAuthConfig
{
    std::string basedir        = "/var/lib/pam_facial_auth";

    std::string device         = "/dev/video0";
    bool fallback_device       = true;

    int width                  = 640;
    int height                 = 480;
    int frames                 = 15;
    int sleep_ms               = 0;

    bool debug                 = false;
    bool verbose               = false;
    bool nogui                 = true;

    std::string image_format   = "jpg";
    bool save_failed_images    = false;
    bool ignore_failure        = false;

    std::string detector_profile     = "";
    std::string recognizer_profile   = "";

    std::string training_method      = "auto";
    bool force_overwrite             = false;

    double lbph_threshold     = 80.0;
    int    eigen_components   = 80;
    double eigen_threshold    = 4000.0;
    int    fisher_components  = 80;
    double fisher_threshold   = 300.0;

    std::string sface_model;
    std::string sface_model_int8;

    std::string haar_cascade_path;

    std::string yunet_model;
    std::string yunet_model_int8;

    std::string dnn_backend = "cpu";
    std::string dnn_target  = "cpu";

    double sface_fp32_threshold = 0.55;
    double sface_int8_threshold = 0.55;

    std::map<std::string,std::string> recognizer_models;
};

// ==========================================================
// DetectorWrapper: uniform face detector abstraction
// ==========================================================
struct DetectorWrapper
{
    enum Type {
        DET_NONE = 0,
        DET_HAAR,
        DET_YUNET
    };

    Type type = DET_NONE;
    bool debug = false;

    std::string model_path;

    cv::CascadeClassifier haar;
    cv::Ptr<cv::dnn::Net> yunet;

    bool detect(const cv::Mat &frame, cv::Rect &face);
};

// ==========================================================
// Public API
// ==========================================================

// Load config file into FacialAuthConfig
bool fa_load_config(
    FacialAuthConfig &cfg,
    std::string &logbuf,
    const std::string &path=""
);

// Compute paths
std::string fa_user_image_dir(const FacialAuthConfig &cfg, const std::string &user);
std::string fa_user_model_path(const FacialAuthConfig &cfg, const std::string &user);

// Capture a batch of frames and save them
bool fa_capture_images(
    const std::string &user,
    const FacialAuthConfig &cfg,
    const std::string &format,
    std::string &log
);

// Train user model (classic or SFace auto-selected)
bool fa_train_user(
    const std::string &user,
    const FacialAuthConfig &cfg,
    std::string &log
);

// Delete user model
bool fa_delete_user(
    const std::string &user,
    const FacialAuthConfig &cfg,
    std::string &log
);

// Authentication test
bool fa_test_user(
    const std::string &,
    const FacialAuthConfig &cfg,
    const std::string &modelPath,
    double &best_conf,
    int &best_label,
    std::string &log,
    double threshold_override = -1.0
);

bool fa_check_root(const std::string &tool_name);
bool fa_file_exists(const std::string &path);


// ==========================================================
// C ABI wrappers for PAM / CLI calls
// ==========================================================
#ifdef __cplusplus
extern "C" {
    #endif

    bool facialauth_load_config(FacialAuthConfig *cfg, const char *path, char *logbuf, int logbuflen);
    bool facialauth_train(const char *user, const FacialAuthConfig *cfg, char *logbuf, int logbuflen);
    bool facialauth_test(
        const char *user,
        const FacialAuthConfig *cfg,
        const char *modelPath,
        double *best_conf,
        int *best_label,
        char *logbuf,
        int logbuflen
    );
    bool facialauth_delete(const char *user, const FacialAuthConfig *cfg, char *logbuf, int logbuflen);

    #ifdef __cplusplus
} // extern "C"
#endif

