/*
 * Project: pam_facial_auth
 * License: GPL-3.0
 */

#ifndef LIBFACIALAUTH_H
#define LIBFACIALAUTH_H

#include <string>
#include <vector>

#ifdef HAVE_OPENCV
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#endif

struct FacialAuthConfig {
    std::string basedir = "/var/lib/pam_facial_auth";
    std::string cascade_path = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml";
    double threshold = 80.0;
    int frames = 20;
    int width = 200;
    int height = 200;
    bool debug = false;
    bool verbose = false;
};

// Funzioni di sistema e utilità
bool fa_check_root(const std::string& tool_name);
bool fa_file_exists(const std::string& path);
bool fa_load_config(FacialAuthConfig& cfg, std::string& log, const std::string& path = "/etc/security/pam_facial_auth.conf");
std::string fa_user_model_path(const FacialAuthConfig& cfg, const std::string& user);

// Gestione catture e training
bool fa_clean_captures(const std::string& user, const FacialAuthConfig& cfg, std::string& log);
bool fa_capture_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& detector, std::string& log);
bool fa_train_user(const std::string& user, const FacialAuthConfig& cfg, std::string& log);

// Riconoscimento
bool fa_test_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& model_path, double& confidence, int& label, std::string& log);

#endif // LIBFACIALAUTH_H
