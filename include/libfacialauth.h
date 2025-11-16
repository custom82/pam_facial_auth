#pragma once
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

// ==========================================
// Configuration
// ==========================================
struct FacialAuthConfig {
    std::string basedir = "/etc/pam_facial_auth";   // base directory for images/models
    std::string device = "/dev/video0";             // default camera
    int width = 640;                                // camera width
    int height = 480;                               // camera height
    double threshold = 80.0;                        // recognition threshold
    int timeout = 5;                                // timeout in seconds
    bool nogui = false;                             // disable GUI preview
    bool debug = false;                             // enable verbose logging
    int frames = 5;                                 // frames to capture for training
    bool fallback_device = true;                    // try /dev/video1 if /dev/video0 fails
    int sleep_ms = 100;                             // delay between frames

    // NEW fields for configuration file support
    std::string model_path;                         // path to trained model file
    std::string haar_cascade_path;                  // cascade XML for detection
    std::string training_method = "lbph";           // algorithm type
    std::string log_file = "/var/log/pam_facial_auth.log"; // path to log file
    bool force_overwrite = false;                   // overwrite existing models
    std::string face_detection_method = "haar";     // method for detection (haar, dnn, etc.)
};

// ==========================================
// API Prototypes
// ==========================================

// Read key=value config file
bool read_kv_config(const std::string &path, FacialAuthConfig &cfg, std::string *logbuf);

// Utility helpers
std::string trim(const std::string &s);
bool str_to_bool(const std::string &s, bool defval);
void ensure_dirs(const std::string &path);
bool file_exists(const std::string &path);
std::string join_path(const std::string &a, const std::string &b);
void sleep_ms(int ms);
void log_tool(bool debug, const char* level, const char* fmt, ...);

// Camera helpers
bool open_camera(const FacialAuthConfig &cfg, cv::VideoCapture &cap, std::string &device_used);

// Path helpers
std::string fa_user_image_dir(const FacialAuthConfig &cfg, const std::string &user);
std::string fa_user_model_path(const FacialAuthConfig &cfg, const std::string &user);

// ==========================================
// Face Recognition Wrapper
// ==========================================
class FaceRecWrapper {
public:
    explicit FaceRecWrapper(const std::string &modelType_ = "LBPH");
    bool Load(const std::string &modelFile);
    bool Save(const std::string &modelFile) const;
    bool Train(const std::vector<cv::Mat> &images, const std::vector<int> &labels);
    bool Predict(const cv::Mat &face, int &prediction, double &confidence) const;
    bool DetectFace(const cv::Mat &frame, cv::Rect &faceROI);

private:
    cv::Ptr<cv::face::FaceRecognizer> recognizer;
    cv::CascadeClassifier faceCascade;
    std::string modelType;
};

// ==========================================
// High-level API
// ==========================================
bool fa_capture_images(const std::string &user,
                       const FacialAuthConfig &cfg,
                       bool force,
                       std::string &log);

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
