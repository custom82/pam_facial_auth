#ifndef LIBFACIALAUTH_H
#define LIBFACIALAUTH_H

#include <string>
#include <map>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/dnn.hpp>

//
// Path di default del file di configurazione
//
#ifndef FACIALAUTH_DEFAULT_CONFIG
#define FACIALAUTH_DEFAULT_CONFIG "/etc/security/pam_facial.conf"
#endif

struct FacialAuthConfig
{
    //
    // PATH / DEVICE
    //
    std::string basedir;
    std::string device;

    //
    // Parametri video
    //
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

    //
    // Profili detector/recognizer
    //
    std::string detector_profile;
    std::string recognizer_profile;

    //
    // Thresholds classici
    //
    double lbph_threshold   = 80.0;
    double eigen_threshold  = 5000.0;
    double fisher_threshold = 500.0;

    int eigen_components  = 80;
    int fisher_components = 80;

    //
    // SFace Thresholds
    //
    double sface_threshold      = 0.5;
    double sface_fp32_threshold = 0.5;
    double sface_int8_threshold = 0.5;

    //
    // Modelli legacy
    //
    std::string model_path;
    std::string haar_cascade_path;

    //
    // Modelli YUNet
    //
    std::string yunet_model;
    std::string yunet_model_int8;

    //
    // Modelli SFace
    //
    std::string sface_model;
    std::string sface_model_int8;

    //
    // Metodo di training: "lbph", "sface", "auto" …
    //
    std::string training_method = "lbph";

    //
    // DNN backend / target
    //
    std::string dnn_backend = "cpu"; // cpu / cuda / cuda_fp16 / opencl ...
    std::string dnn_target  = "cpu"; // cpu / cuda / cuda_fp16 / opencl ...

    //
    // Mappature dinamiche config: detect_*  e recognize_*
    //
    std::map<std::string, std::string> detector_models;
    std::map<std::string, std::string> recognizer_models;

    //
    // Output images
    //
    std::string image_format = "jpg";
};

struct DetectorWrapper
{
    //
    // Tipo di detector
    //
    enum DetectorType {
        DET_NONE = 0,
        DET_HAAR,
        DET_YUNET
    };

    DetectorType type = DET_NONE;

    // Haar cascade
    cv::CascadeClassifier haar;

    // YuNet
    cv::Size input_size = cv::Size(320, 320);
    cv::Ptr<cv::dnn::Net> yunet;   // YuNet come cv::Ptr
    std::string model_path;

    bool debug = false;            // flag di debug

    // API di rilevazione volto
    bool detect(const cv::Mat &frame, cv::Rect &face) const;
};


// --------- FUNZIONI DELLA LIBRERIA ---------

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
