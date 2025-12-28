/*
 * Project: pam_facial_auth
 * License: GPL-3.0
 */

#ifndef LIBFACIALAUTH_H
#define LIBFACIALAUTH_H

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

#define FA_EXPORT __attribute__((visibility("default")))

struct FacialAuthConfig {
    std::string basedir;
    std::string device;
    std::string recognize_sface; // Mappato su recognize_sface nel conf
    std::string detect_yunet;    // Mappato su detect_yunet nel conf
    std::string cascade_path;    // Aggiunto per plugin_sface.cpp
    std::string detector;        // Aggiunto per facial_capture.cpp
    std::string method;

    double threshold = 0.0;      // Aggiunto per pam_facial_auth.cpp e facial_test.cpp
    double sface_threshold = 0.0;
    double lbph_threshold = 0.0;

    int frames = 0;
    int width = 0;
    int height = 0;
    int sleep_ms = 0;
    double capture_delay = 0.0;  // Aggiunto per facial_capture.cpp

    bool debug = false;
    bool verbose = false;
    bool nogui = false;
};

class RecognizerPlugin {
public:
    virtual ~RecognizerPlugin() = default;
    virtual bool load(const std::string& path) = 0;
    virtual bool train(const std::vector<cv::Mat>& faces, const std::vector<int>& labels, const std::string& save_path) = 0;
    virtual bool predict(const cv::Mat& face, int& label, double& confidence) = 0;
    virtual std::string get_name() const = 0;
};

extern "C" {
    FA_EXPORT bool fa_check_root(const std::string& tool_name);
    FA_EXPORT bool fa_load_config(FacialAuthConfig& cfg, std::string& log, const std::string& path = "/etc/security/pam_facial_auth.conf");
    FA_EXPORT std::string fa_user_model_path(const FacialAuthConfig& cfg, const std::string& user);
    FA_EXPORT bool fa_clean_captures(const std::string& user, const FacialAuthConfig& cfg, std::string& log);
    FA_EXPORT bool fa_capture_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& device_path, std::string& log);
    FA_EXPORT bool fa_train_user(const std::string& user, const FacialAuthConfig& cfg, std::string& log);
    FA_EXPORT bool fa_test_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& model_path, double& confidence, int& label, std::string& log);
}

#endif
