/*
 * Project: pam_facial_auth
 * License: GPL-3.0
 */

#ifndef LIBFACIALAUTH_H
#define LIBFACIALAUTH_H

#include <string>
#include <vector>
#include <memory>

#ifdef HAVE_OPENCV
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#endif

struct FacialAuthConfig {
    std::string basedir = "/var/lib/pam_facial_auth";
    std::string cascade_path = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml";
    std::string detector = "haarcascade";
    std::string method = "lbph"; // Metodo di riconoscimento
    double threshold = 80.0;
    int frames = 20;
    int width = 200;
    int height = 200;
    double capture_delay = 0.1;
    bool debug = false;
    bool verbose = false;
    bool nogui = false;
};

// Interfaccia per i plugin di riconoscimento
class RecognizerPlugin {
public:
    virtual ~RecognizerPlugin() = default;
    virtual bool load(const std::string& path) = 0;
    virtual bool train(const std::vector<cv::Mat>& faces, const std::vector<int>& labels, const std::string& save_path) = 0;
    virtual bool predict(const cv::Mat& face, int& label, double& confidence) = 0;
    virtual std::string get_name() const = 0;
};

// Funzioni di utilità
bool fa_check_root(const std::string& tool_name);
bool fa_load_config(FacialAuthConfig& cfg, std::string& log, const std::string& path = "/etc/security/pam_facial_auth.conf");
std::string fa_user_model_path(const FacialAuthConfig& cfg, const std::string& user);

// Gestione dati
bool fa_clean_captures(const std::string& user, const FacialAuthConfig& cfg, std::string& log);
bool fa_capture_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& device_path, std::string& log);

// Training e Test (Aggiornati per supportare i metodi/plugin)
bool fa_train_user(const std::string& user, const FacialAuthConfig& cfg, std::string& log);
bool fa_test_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& model_path, double& confidence, int& label, std::string& log);

#endif
