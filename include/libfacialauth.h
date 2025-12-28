/*
 * Project: pam_facial_auth
 * License: GPL-3.0
 */

#ifndef LIBFACIALAUTH_H
#define LIBFACIALAUTH_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#ifdef __cplusplus
extern "C" {
    #endif

    struct FacialAuthConfig {
        std::string basedir = "/var/lib/pam_facial_auth";
        std::string modeldir = "/etc/security/pam_facial_auth";
        std::string device = "/dev/video0";
        std::string detect_yunet;
        std::string recognize_sface;
        std::string cascade_path;
        std::string detector = "none";
        std::string method = "auto";
        std::string image_format = "jpg";

        double threshold = 0.0;
        int frames = 30;
        int width = 640;
        int height = 480;
        double capture_delay = 0.1;

        bool debug = false;
        bool verbose = false;
        bool nogui = false;
    };

    // Funzioni esportate
    bool fa_check_root(const std::string& tool_name);
    bool fa_load_config(FacialAuthConfig& cfg, std::string& log, const std::string& path);
    std::string fa_user_model_path(const FacialAuthConfig& cfg, const std::string& user);
    bool fa_clean_captures(const std::string& user, const FacialAuthConfig& cfg, std::string& log);
    bool fa_capture_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& device_path, std::string& log);
    bool fa_train_user(const std::string& user, const FacialAuthConfig& cfg, std::string& log);
    bool fa_test_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& model_path, double& confidence, int& label, std::string& log);

    #ifdef __cplusplus
}
#endif

#endif
