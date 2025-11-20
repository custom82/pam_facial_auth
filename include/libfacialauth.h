#ifndef LIBFACIALAUTH_H
#define LIBFACIALAUTH_H

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

// ==========================================================
// Default config file path
// ==========================================================
#ifndef FACIALAUTH_CONFIG_DEFAULT
#define FACIALAUTH_CONFIG_DEFAULT "/etc/security/pam_facial.conf"
#endif

// ==========================================================
// Extra utility helpers
// ==========================================================

// Valid image formats: .jpg .jpeg .png
bool fa_is_valid_image(const std::string &path);


// ==========================================================
// Global configuration structure
// ==========================================================
struct FacialAuthConfig {
    std::string basedir = "/etc/pam_facial_auth";
    std::string device = "/dev/video0";
    std::string model_path;
    std::string haar_cascade_path = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml";
    std::string training_method = "lbph";
    std::string face_detection_method = "haar";
    std::string log_file = "/var/log/pam_facial_auth.log";

    int width = 640;
    int height = 480;
    int frames = 5;
    int timeout = 5;
    int sleep_ms = 500;

    double threshold = 80.0;

    bool debug = false;
    bool nogui = false;
    bool fallback_device = true;
    bool force_overwrite = false;
    bool ignore_failure = false;
};

// ==========================================================
// Utility functions
// ==========================================================
std::string trim(const std::string &s);
bool str_to_bool(const std::string &s, bool defval);
bool read_kv_config(const std::string &path, FacialAuthConfig &cfg, std::string *logbuf);
void ensure_dirs(const std::string &path);
bool file_exists(const std::string &path);
std::string join_path(const std::string &a, const std::string &b);
void sleep_ms(int ms);

// Logging
void log_tool(const FacialAuthConfig &cfg, const char *level, const char *fmt, ...);

// Camera / path helpers
bool open_camera(const FacialAuthConfig &cfg, cv::VideoCapture &cap, std::string &device_used);
std::string fa_user_image_dir(const FacialAuthConfig &cfg, const std::string &user);
std::string fa_user_model_path(const FacialAuthConfig &cfg, const std::string &user);

// ==========================================================
// OpenCV FaceRecognizer wrapper
// ==========================================================
class FaceRecWrapper {
public:
    explicit FaceRecWrapper(const std::string &modelType_ = "lbph");

    bool Load(const std::string &modelFile);
    bool Save(const std::string &modelFile) const;
    bool Train(const std::vector<cv::Mat> &images, const std::vector<int> &labels);
    bool Predict(const cv::Mat &face, int &prediction, double &confidence) const;
    bool DetectFace(const cv::Mat &frame, cv::Rect &faceROI);

private:
    std::string modelType;
    cv::Ptr<cv::face::FaceRecognizer> recognizer;
    cv::CascadeClassifier faceCascade;
};

// ==========================================================
// High-level API
// ==========================================================
bool fa_capture_images(const std::string &user,
                       const FacialAuthConfig &cfg,
                       bool force,
                       std::string &log,
                       const std::string &img_format = "png");

bool fa_train_user(const std::string &user,
                   const FacialAuthConfig &cfg,
                   const std::string &method,
                   const std::string &inputDir,
                   const std::string &outputModel,
                   bool force,
                   std::string &log);

bool fa_test_user(const std::string &user,
                  const FacialAuthConfig &cfg,
                  const std::string &modelPath,
                  double &best_conf,
                  int &best_label,
                  std::string &log);

// Maintenance helpers and root check
bool fa_clean_images(const FacialAuthConfig &cfg, const std::string &user);
bool fa_clean_model(const FacialAuthConfig &cfg, const std::string &user);
void fa_list_images(const FacialAuthConfig &cfg, const std::string &user);
bool fa_check_root(const char *tool_name);

#endif // LIBFACIALAUTH_H
