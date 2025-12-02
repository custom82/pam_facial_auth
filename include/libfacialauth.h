#ifndef LIBFACIALAUTH_H
#define LIBFACIALAUTH_H

#include <string>
#include <map>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/dnn.hpp>

// Default config path
#ifndef FACIALAUTH_DEFAULT_CONFIG
#define FACIALAUTH_DEFAULT_CONFIG "/etc/security/pam_facial.conf"
#endif

//
// Main configuration structure
//
struct FacialAuthConfig
{
    // Paths and device
    std::string basedir;
    std::string device;

    // Video parameters
    int width  = 640;
    int height = 480;

    int frames   = 20;
    int sleep_ms = 200;

    bool fallback_device    = false;
    bool debug              = false;
    bool verbose            = false;
    bool nogui              = false;
    bool force_overwrite    = false;
    bool ignore_failure     = false;
    bool save_failed_images = false;

    // Model profile selectors
    std::string detector_profile;
    std::string recognizer_profile;

    // Classic face recognizer thresholds
    double lbph_threshold   = 80.0;
    double eigen_threshold  = 5000.0;
    double fisher_threshold = 500.0;

    int eigen_components  = 80;
    int fisher_components = 80;

    // SFace thresholds
    double sface_threshold      = 0.5;
    double sface_fp32_threshold = 0.5;
    double sface_int8_threshold = 0.5;

    // Legacy models
    std::string model_path;
    std::string haar_cascade_path;

    // YuNet models
    std::string yunet_model;
    std::string yunet_model_int8;

    // SFace models
    std::string sface_model;
    std::string sface_model_int8;

    // Training method selector: lbph / sface / auto
    std::string training_method = "lbph";

    // DNN control
    std::string dnn_backend = "cpu";
    std::string dnn_target  = "cpu";

    // Dynamic config model maps
    std::map<std::string, std::string> detector_models;
    std::map<std::string, std::string> recognizer_models;

    // Output format
    std::string image_format = "jpg";
};


//
// Unified face detector wrapper:
// Supports HAAR and YuNet detection.
//
struct DetectorWrapper
{
    enum DetectorType {
        DET_NONE = 0,
        DET_HAAR,
        DET_YUNET
    };

    DetectorType type = DET_NONE;

    cv::CascadeClassifier haar;
    cv::Ptr<cv::dnn::Net> yunet;
    cv::Size input_size = cv::Size(320,320);

    bool debug = false;
    std::string model_path;

    // Unified detector interface
    bool detect(const cv::Mat &frame, cv::Rect &face);
};


//
// SFace model resolution function
//
bool resolve_sface_model(
    const FacialAuthConfig &cfg,
    const std::string &profile,
    std::string &out_model_file,
    std::string &out_resolved_profile
);

//
// SFace embedding computation
//
bool compute_sface_embedding(
    const FacialAuthConfig &cfg,
    const cv::Mat &face,
    const std::string &profile,
    cv::Mat &embedding,
    std::string &log
);


//
// Main library API
//
bool fa_load_config(FacialAuthConfig &cfg,
                    std::string &log,
                    const std::string &path);

std::string fa_user_image_dir(const FacialAuthConfig &cfg,
                              const std::string &user);

std::string fa_user_model_path(const FacialAuthConfig &cfg,
                               const std::string &user);

bool fa_capture_images(const std::string &user,
                       const FacialAuthConfig &cfg,
                       const std::string &format,
                       std::string &log);

bool fa_train_user(const std::string &user,
                   const FacialAuthConfig &cfg,
                   std::string &log);

bool fa_test_user(const std::string &user,
                  const FacialAuthConfig &cfg,
                  const std::string &model_path,
                  double &best_conf,
                  int &best_label,
                  std::string &log,
                  double threshold_override = -1.0);

bool fa_check_root(const std::string &tool_name);

#endif // LIBFACIALAUTH_H
