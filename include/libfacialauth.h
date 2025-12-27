#ifndef LIBFACIALAUTH_H
#define LIBFACIALAUTH_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

// L'unico path ammesso: il punto di ingresso della configurazione
#ifndef FACIALAUTH_DEFAULT_CONFIG
#define FACIALAUTH_DEFAULT_CONFIG "/etc/pam_facial_auth.conf"
#endif

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
    std::string training_method = "lbph";

    // Questi verranno popolati ESCLUSIVAMENTE dal file di config o defaults dinamici
    std::string basedir;
    std::string cascade_path;

    double threshold = 80.0;
    double lbph_threshold = 80.0;
    double sface_threshold = 0.36;
};

// API
bool fa_check_root(const std::string& tool_name);
bool fa_file_exists(const std::string& path);
bool fa_load_config(FacialAuthConfig& cfg, std::string& log, const std::string& path);
std::string fa_user_model_path(const FacialAuthConfig& cfg, const std::string& user);
bool fa_capture_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& detector, std::string& log);
bool fa_train_user(const std::string& user, const FacialAuthConfig& cfg, std::string& log);
bool fa_test_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& model_path, double& confidence, int& label, std::string& log);

#endif
