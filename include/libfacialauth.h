#ifndef LIB_FACIALAUTH_H
#define LIB_FACIALAUTH_H

#include <string>
#include <vector>
#include <opencv2/core.hpp>

#define FACIALAUTH_DEFAULT_CONFIG "/etc/security/pam_facial.conf"

struct FacialAuthConfig {
    std::string basedir;
    std::string device;
    std::string training_method;
    std::string image_format;

    // Paths populated ONLY via config file
    std::string detect_model_path;
    std::string rec_model_path;

    int frames = 30;
    int width = 1280;
    int height = 720;
    bool force = false;
    bool nogui = true;
    bool debug = false;

    double sface_threshold = 0.36;
    double lbph_threshold = 60.0;
};

class RecognizerPlugin {
public:
    virtual ~RecognizerPlugin() = default;
    virtual bool load(const std::string& path) = 0;
    virtual bool train(const std::vector<cv::Mat>& faces, const std::vector<int>& labels, const std::string& save_path) = 0;
    virtual bool predict(const cv::Mat& face, int& label, double& confidence) = 0;
};

bool fa_load_config(FacialAuthConfig &cfg, std::string &log, const std::string &path);
bool fa_check_root(const std::string &tool_name);
bool fa_delete_user_data(const std::string &user, const FacialAuthConfig &cfg);
std::string fa_user_model_path(const FacialAuthConfig &cfg, const std::string &user);
bool fa_capture_user(const std::string &user, const FacialAuthConfig &cfg, const std::string &det_type, std::string &log);
bool fa_train_user(const std::string &user, const FacialAuthConfig &cfg, std::string &log);
bool fa_test_user(const std::string &user, const FacialAuthConfig &cfg, const std::string &model_path, double &conf, int &label, std::string &log);

#endif
