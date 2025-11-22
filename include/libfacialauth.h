#ifndef LIBFACIALAUTH_H
#define LIBFACIALAUTH_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/face.hpp>
#include <string>

struct FacialAuthConfig {
    std::string basedir;
    std::string device;
    int camera_index;
    int frames;
    int width;
    int height;
    int sleep_ms;
    double threshold;
    bool debug;
    bool force_overwrite;

    // DNN settings
    std::string dnn_type;
    std::string dnn_model_path;
    std::string dnn_proto_path;
    std::string dnn_device;
    double dnn_threshold;
    std::string dnn_profile;

    // DNN model paths for various profiles
    std::string dnn_model_fast;
    std::string dnn_model_sface;
    std::string dnn_model_lresnet100;
    std::string dnn_model_openface;

    std::string dnn_model_yunet;
    std::string dnn_model_detector_caffe;
    std::string dnn_model_detector_fp16;
    std::string dnn_model_detector_uint8;
    std::string dnn_proto_detector_caffe;

    std::string dnn_model_emotion;
    std::string dnn_model_keypoints;
    std::string dnn_model_face_landmark_tflite;
    std::string dnn_model_face_detection_tflite;
    std::string dnn_model_face_blendshapes_tflite;

    // Haar cascade path
    std::string haar_cascade;
};

// Funzioni per il caricamento della configurazione
bool fa_load_config(const std::string &path, FacialAuthConfig &cfg, std::string &log);

// Funzione per ottenere il percorso della directory delle immagini utente
std::string fa_user_image_dir(const FacialAuthConfig &cfg, const std::string &user);

// Funzione per ottenere il percorso del modello utente
std::string fa_user_model_path(const FacialAuthConfig &cfg, const std::string &user);

// Funzione per il rilevamento del volto utilizzando il DNN o la cascata Haar
bool fa_select_dnn_profile(FacialAuthConfig &cfg, const std::string &profile, std::string &log);
bool fa_detect_face(const cv::Mat &frame, cv::Rect &faceROI, const FacialAuthConfig &cfg);

// Funzioni di utilità
bool file_exists(const std::string &path);
void ensure_dirs(const std::string &path);
std::string join_path(const std::string &a, const std::string &b);
void log_tool(const FacialAuthConfig &cfg, const char *level, const char *fmt, ...);

#endif // LIBFACIALAUTH_H
