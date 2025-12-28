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

// Configurazione globale
struct FacialAuthConfig {
    std::string basedir = "/var/lib/pam_facial_auth";
    std::string modeldir = "/etc/security/pam_facial_auth";
    std::string device = "/dev/video0";
    std::string method = "auto";
    std::string image_format = "jpg";
    double threshold = 0.0;
    int frames = 30;
    int width = 640;
    int height = 480;
    bool debug = false;
    bool verbose = false;
    bool nogui = false;
};

// Classe base per i plugin (Risolve l'errore 'expected class-name')
class RecognizerPlugin {
public:
    virtual ~RecognizerPlugin() = default;
    virtual bool load(const std::string& path) = 0;
    virtual bool train(const std::vector<cv::Mat>& faces, const std::vector<int>& labels, const std::string& save_path) = 0;
    virtual bool predict(const cv::Mat& face, int& label, double& confidence) = 0;
    virtual std::string get_name() const = 0;
};

#ifdef __cplusplus
extern "C" {
    #endif

    // Funzioni della libreria
    bool fa_check_root(const std::string& tool_name);

    // Il parametro 'path' è opzionale (Risolve 'too few arguments' in pam_facial_auth.cpp)
    bool fa_load_config(FacialAuthConfig& cfg, std::string& log, const std::string& path = "");

    std::string fa_user_model_path(const FacialAuthConfig& cfg, const std::string& user);
    bool fa_clean_captures(const std::string& user, const FacialAuthConfig& cfg, std::string& log);
    bool fa_capture_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& device_path, std::string& log);
    bool fa_train_user(const std::string& user, const FacialAuthConfig& cfg, std::string& log);
    bool fa_test_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& model_path, double& confidence, int& label, std::string& log);

    #ifdef __cplusplus
}
#endif

#endif
