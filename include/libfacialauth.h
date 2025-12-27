#ifndef LIBFACIALAUTH_H
#define LIBFACIALAUTH_H

#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <memory>

#define FACIALAUTH_DEFAULT_CONFIG "/etc/security/pam_facial.conf"

struct FacialAuthConfig {
    std::string basedir = "/var/lib/pam_facial_auth";
    std::string device = "0";
    std::string training_method = "lbph"; // lbph, eigen, fisher, sface
    std::string image_format = "jpg";
    std::string recognize_sface = "/usr/share/pam_facial_auth/models/face_recognition_sface_2021dec.onnx";
    std::string detect_model_path = "/usr/share/pam_facial_auth/models/face_detection_yunet.onnx";

    int width = 640;
    int height = 480;
    int frames = 50;
    int sleep_ms = 100;
    bool debug = false;
    bool verbose = false;
    bool nogui = false;
    bool force = false;

    double lbph_threshold = 80.0;
    double sface_threshold = 0.36;
};

// Interfaccia Plugin
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
bool fa_capture_user(const std::string &user, const FacialAuthConfig &cfg, const std::string &detector_type, std::string &log);
bool fa_train_user(const std::string &user, const FacialAuthConfig &cfg, std::string &log);
bool fa_test_user(const std::string &user, const FacialAuthConfig &cfg, const std::string &modelPath, double &best_conf, int &best_label, std::string &log);
bool fa_test_user_interactive(const std::string &user, const FacialAuthConfig &cfg, std::string &log);

std::unique_ptr<RecognizerPlugin> fa_create_plugin(const FacialAuthConfig& cfg);
std::string fa_user_model_path(const FacialAuthConfig &cfg, const std::string &user);
bool fa_check_root(const std::string &tool_name);
bool fa_file_exists(const std::string &path);

#endif
