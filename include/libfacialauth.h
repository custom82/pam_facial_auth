/*
 * Project: pam_facial_auth
 * License: GPL-3.0
 */

#ifndef LIBFACIALAUTH_H
#define LIBFACIALAUTH_H

#include <string>
#include <vector>

// Conditionally include OpenCV headers based on CMake definition
#ifdef HAVE_OPENCV
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#endif

/**
 * The only hardcoded path allowed in the source code.
 * It follows the Linux PAM security standards.
 */
#ifndef FACIALAUTH_DEFAULT_CONFIG
#define FACIALAUTH_DEFAULT_CONFIG "/etc/security/pam_facial_auth.conf"
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

    // These paths must be populated from the config file at runtime
    std::string basedir;
    std::string cascade_path;

    double threshold = 80.0;
    double lbph_threshold = 80.0;
    double sface_threshold = 0.36;
};

// API Declarations (GPL-3.0)

// Check if the current user has root privileges
bool fa_check_root(const std::string& tool_name);

// Utility to check if a file exists on the filesystem
bool fa_file_exists(const std::string& path);

// Load the configuration from /etc/security/pam_facial_auth.conf
bool fa_load_config(FacialAuthConfig& cfg, std::string& log, const std::string& path);

// Generate the dynamic path for a specific user's XML model
std::string fa_user_model_path(const FacialAuthConfig& cfg, const std::string& user);

// Facial recognition operations (Wrapped in HAVE_OPENCV for safety)
bool fa_capture_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& detector, std::string& log);
bool fa_train_user(const std::string& user, const FacialAuthConfig& cfg, std::string& log);
bool fa_test_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& model_path, double& confidence, int& label, std::string& log);

#endif // LIBFACIALAUTH_H
