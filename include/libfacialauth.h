#ifndef LIBFACIALAUTH_H
#define LIBFACIALAUTH_H

#include <string>
#include <vector>
#include <opencv2/core.hpp>

#define FACIALAUTH_DEFAULT_CONFIG "/etc/security/pam_facial.conf"

struct FacialAuthConfig {
    std::string basedir = "/var/lib/pam_facial_auth";
    std::string device = "0";
    std::string training_method = "lbph";
    std::string image_format = "jpg";
    std::string detect_model_path = "/usr/share/pam_facial_auth/models/face_detection_yunet.onnx";
    int frames = 50;
    bool force = false;
    bool nogui = false;
    double lbph_threshold = 80.0;
};

// Interfaccia base per i plugin
class RecognizerPlugin {
public:
    virtual ~RecognizerPlugin() = default;
    virtual bool load(const std::string& path) = 0;
    virtual bool train(const std::vector<cv::Mat>& faces, const std::vector<int>& labels, const std::string& save_path) = 0;
    virtual bool predict(const cv::Mat& face, int& label, double& confidence) = 0;
};

// API della libreria
bool fa_load_config(FacialAuthConfig &cfg, std::string &log, const std::string &path);
bool fa_check_root(const std::string &tool_name);
bool fa_file_exists(const std::string &path);
std::string fa_user_model_path(const FacialAuthConfig &cfg, const std::string &user);
bool fa_capture_user(const std::string &user, const FacialAuthConfig &cfg, const std::string &det_type, std::string &log);
bool fa_train_user(const std::string &user, const FacialAuthConfig &cfg, std::string &log);
bool fa_test_user(const std::string &user, const FacialAuthConfig &cfg, const std::string &model_path, double &conf, int &label, std::string &log);

#endif
