#ifndef LIBFACIALAUTH_H
#define LIBFACIALAUTH_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

// Macro richiesta dai tool
#define FACIALAUTH_DEFAULT_CONFIG "/etc/pam_facial_auth.conf"

struct FacialAuthConfig {
    bool verbose = false;
    bool debug = false;
    bool force = false;
    bool nogui = true;

    int sleep_ms = 100;
    int frames = 50;
    int width = 640;
    int height = 480;

    std::string image_format = "jpg";
    std::string model_type = "lbph";
    std::string training_method = "lbph"; // Può essere "lbph" o "sface"
    std::string basedir = "/var/lib/pam_facial_auth";
    std::string cascade_path = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml";

    double threshold = 80.0;
    double lbph_threshold = 80.0;
    double sface_threshold = 0.36;
};

// --- API della Libreria ---

// Utility di sistema e configurazione
bool fa_check_root(const std::string& tool_name);
bool fa_load_config(FacialAuthConfig& cfg, std::string& log, const std::string& path);
std::string fa_user_model_path(const FacialAuthConfig& cfg, const std::string& user);

// Funzioni di cattura e training
bool fa_capture_dataset(const FacialAuthConfig& cfg, std::string& log, const std::string& user, int count);
bool fa_capture_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& detector, std::string& log);
bool fa_train_user(const std::string& user, const FacialAuthConfig& cfg, std::string& log);

// Funzione di verifica (Parametri allineati alla chiamata del tool)
bool fa_test_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& model_path, double& confidence, int& label, std::string& log);

#endif
