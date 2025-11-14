#pragma once

#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

// -------------------------------------------------------------
// CONFIGURAZIONE COMPLETA
// -------------------------------------------------------------
struct FacialAuthConfig {
    std::string haar_path = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml";
    std::string dnn_proto = "";    // deploy.prototxt
    std::string dnn_model = "";    // res10_300x300.caffemodel

    int frame_width  = 640;
    int frame_height = 480;

    int sleep_ms = 100;

    bool debug = false;
    bool nogui = false;
};

// -------------------------------------------------------------
// API DELLA LIBRERIA
// -------------------------------------------------------------

bool load_detectors(
    const FacialAuthConfig &cfg,
    cv::CascadeClassifier &haar,
    cv::dnn::Net &dnn,
    bool &use_dnn,
    std::string &log
);

bool detect_face(
    const FacialAuthConfig &cfg,
    const cv::Mat &frame,
    cv::Rect &face_roi,
    cv::CascadeClassifier &haar,
    cv::dnn::Net &dnn
);

// -------------------------------------------------------------
// WRAPPER PER CAPTURE / TEST
// -------------------------------------------------------------

class FaceRecWrapper {
public:

    // cattura immagini per l’utente
    bool CaptureImages(const std::string &user, const FacialAuthConfig &cfg);

    // usa SOLO Haar/DNN gestito internamente dalla lib
    bool DetectFace(const cv::Mat &frame, cv::Rect &faceROI);
};
