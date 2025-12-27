#pragma once
#include <string>
#include <vector>
#include <memory>
#include <opencv2/core.hpp>

#ifndef FACIALAUTH_DEFAULT_CONFIG
#define FACIALAUTH_DEFAULT_CONFIG "/etc/security/pam_facial.conf"
#endif

struct FacialAuthConfig {
    std::string basedir = "/var/lib/pam_facial_auth";
    std::string device = "/dev/video0";
    int width = 1280; int height = 720;
    int frames = 30; int sleep_ms = 100;
    bool debug = false; bool ignore_failure = false;
    std::string image_format = "jpg";
    std::string training_method = "auto";
    std::string detect_yunet = "";
    std::string recognize_sface = "";
    double sface_threshold = 0.36;
    double lbph_threshold = 60.0;
    double eigen_threshold = 5000.0;
    double fisher_threshold = 500.0;
    int eigen_components = 80;
    int fisher_components = 80;
};

class RecognizerPlugin {
public:
    virtual ~RecognizerPlugin() = default;
    virtual bool load(const std::string& path) = 0;
    virtual bool train(const std::vector<cv::Mat>& faces, const std::vector<int>& labels, const std::string& save_path) = 0;
    virtual bool predict(const cv::Mat& face, int& label, double& confidence) = 0;
    virtual std::string get_name() const = 0;
};

bool fa_load_config(FacialAuthConfig &cfg, std::string &log, const std::string &path);
bool fa_train_user(const std::string &user, const FacialAuthConfig &cfg, std::string &log);
bool fa_test_user(const std::string &user, const FacialAuthConfig &cfg, const std::string &modelPath,
                  double &best_conf, int &best_label, std::string &log, double threshold_override = -1.0);
std::string fa_user_model_path(const FacialAuthConfig &cfg, const std::string &user);
bool fa_check_root(const std::string &tool_name);
bool fa_file_exists(const std::string &path);
