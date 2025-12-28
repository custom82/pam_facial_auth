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

// Macro per la visibilità dei simboli su Linux
#define FA_EXPORT __attribute__((visibility("default")))

struct FacialAuthConfig {
    std::string basedir = "/var/lib/pam_facial_auth";
    std::string cascade_path = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml";
    std::string recognize_sface = "/usr/share/opencv4/networks/face_recognizer_fast.onnx";
    std::string method = "lbph";
    double threshold = 80.0;
    double sface_threshold = 0.36;
    double lbph_threshold = 80.0;
    int frames = 20;
    int width = 200;
    int height = 200;
    double capture_delay = 0.1;
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

// Esportazione funzioni per i tool e il modulo PAM
extern "C" {
    FA_EXPORT bool fa_check_root(const std::string& tool_name);
    FA_EXPORT bool fa_load_config(FacialAuthConfig& cfg, std::string& log, const std::string& path = "/etc/security/pam_facial_auth.conf");
    FA_EXPORT std::string fa_user_model_path(const FacialAuthConfig& cfg, const std::string& user);
    FA_EXPORT bool fa_clean_captures(const std::string& user, const FacialAuthConfig& cfg, std::string& log);
    FA_EXPORT bool fa_capture_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& device_path, std::string& log);
    FA_EXPORT bool fa_train_user(const std::string& user, const FacialAuthConfig& cfg, std::string& log);
    FA_EXPORT bool fa_test_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& model_path, double& confidence, int& label, std::string& log);
}

#endif // LIBFACIALAUTH_H
