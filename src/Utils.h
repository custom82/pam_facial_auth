#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

struct FacialAuthConfig {
    bool debug = false;
    bool nogui = true;
    double threshold = 80.0;
    int timeout = 10;                // secondi
    std::string model_path = "/etc/pam_facial_auth";
    std::string device = "/dev/video0";
    int width = 640;
    int height = 480;
    std::string model = "lbph";      // lbph | eigen | fisher
    std::string detector = "auto";   // auto | haar | dnn
    std::string model_format = "both"; // xml | yaml | onnx | both
    int frames = 20;                 // frame da catturare in training
    bool fallback_device = true;     // prova /dev/video1 se /dev/video0 fallisce
};

std::string trim(const std::string &s);
bool str_to_bool(const std::string &s, bool defval);
bool read_kv_config(const std::string &path, FacialAuthConfig &cfg, std::string *logbuf=nullptr);
void ensure_dirs(const std::string &path);
bool file_exists(const std::string &path);
std::string join_path(const std::string &a, const std::string &b);
void sleep_ms(int ms);

/// Log generico per tool
void log_tool(bool debug, const char* level, const char* fmt, ...);

/// Apertura camera con fallback opzionale
bool open_camera(const FacialAuthConfig &cfg, cv::VideoCapture &cap, std::string &device_used);

/// Face detection (ritorna true se almeno un volto, restituisce ROI del primo)
bool detect_face(
    const FacialAuthConfig &cfg,
    const cv::Mat &frame,
    cv::Rect &face_roi,
    cv::CascadeClassifier &haar,
    cv::dnn::Net &dnn);

/// Carica detector (haar o dnn). Se auto, prova dnn e fallback ad haar.
void load_detectors(const FacialAuthConfig &cfg,
                    cv::CascadeClassifier &haar,
                    cv::dnn::Net &dnn,
                    bool &use_dnn, std::string &log);

#endif
