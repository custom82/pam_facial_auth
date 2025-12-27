#ifndef LIBFACIALAUTH_H
#define LIBFACIALAUTH_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>

#define FACIALAUTH_DEFAULT_CONFIG "/etc/security/pam_facial.conf"

struct FacialAuthConfig {
    std::string basedir = "/var/lib/pam_facial_auth";
    std::string device = "0";
    int width = 640;
    int height = 480;
    int frames = 30;
    int sleep_ms = 100;
    bool debug = false;
    std::string training_method = "auto";
    std::string recognize_sface = "/usr/share/opencv4/dnn/models/face_recognition_sface_2021dec.onnx";
    double sface_threshold = 0.36;
    double lbph_threshold = 80.0;
    std::string image_format = "jpg";
};

class RecognizerPlugin {
public:
    virtual ~RecognizerPlugin() = default;
    virtual bool load(const std::string& path) = 0;
    virtual bool train(const std::vector<cv::Mat>& faces, const std::vector<int>& labels, const std::string& save_path) = 0;
    virtual bool predict(const cv::Mat& face, int& label, double& confidence) = 0;
    virtual std::string get_name() const = 0;
};

// API
bool fa_load_config(FacialAuthConfig &cfg, std::string &log, const std::string &path);
bool fa_train_user(const std::string &user, const FacialAuthConfig &cfg, std::string &log);
bool fa_test_user(const std::string &user, const FacialAuthConfig &cfg, const std::string &modelPath, double &best_conf, int &best_label, std::string &log, double threshold_override = -1.0);
std::string fa_user_model_path(const FacialAuthConfig &cfg, const std::string &user);
bool fa_file_exists(const std::string &path);
bool fa_check_root(const std::string &tool_name);

#endif
