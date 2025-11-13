#ifndef LIBFACIALAUTH_H
#define LIBFACIALAUTH_H

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>      // OpenCV face module
#include <opencv2/objdetect.hpp> // Per il rilevamento facciale
#include <opencv2/dnn.hpp>       // Per i modelli DNN
#include <string>
#include <vector>

// ==========================================================
// Classe per la gestione del riconoscimento facciale base
// ==========================================================
class FaceRecWrapper {
public:
    FaceRecWrapper(const std::string& modelPath, const std::string& name, const std::string& model_type);
    void Train(const std::vector<cv::Mat>& images, const std::vector<int>& labels);
    void Recognize(cv::Mat& face);
    void Load(const std::string& modelFile);
    void Save(const std::string& modelFile);
    void Predict(cv::Mat& face, int& prediction, double& confidence);

    // 🔹 Nuovo metodo per il rilevamento dei volti
    bool DetectFace(const FacialAuthConfig &cfg, const cv::Mat &frame, cv::Rect &faceROI,
                    cv::CascadeClassifier &haar, cv::dnn::Net &dnn);

private:
    cv::Ptr<cv::face::LBPHFaceRecognizer> recognizer;
    cv::CascadeClassifier faceCascade; // Haar cascade per il rilevamento
    std::string modelType;
};

// ==========================================================
// Classe principale per autenticazione facciale PAM
// ==========================================================
class FacialAuth {
public:
    FacialAuth();
    ~FacialAuth();

    bool Authenticate(const std::string &user);

private:
    bool LoadModel(const std::string &modelPath);
    bool TrainModel(const std::vector<cv::Mat> &images, const std::vector<int> &labels);
    bool RecognizeFace(const cv::Mat &faceImage);

    cv::Ptr<cv::face::LBPHFaceRecognizer> recognizer;
    std::string modelPath;
};

// ==========================================================
// Struttura di configurazione globale
// ==========================================================
struct FacialAuthConfig {
    bool debug = false;
    bool nogui = true;
    double threshold = 80.0;
    int timeout = 10;
    std::string model_path = "/etc/pam_facial_auth";
    std::string device = "/dev/video0";
    int width = 640;
    int height = 480;
    std::string model = "lbph";
    std::string detector = "auto";
    std::string model_format = "both";
    int frames = 20;
    bool fallback_device = true;
};

// ==========================================================
// Utility functions (config, I/O, detection helpers)
// ==========================================================
std::string trim(const std::string &s);
bool str_to_bool(const std::string &s, bool defval);
bool read_kv_config(const std::string &path, FacialAuthConfig &cfg, std::string *logbuf=nullptr);
void ensure_dirs(const std::string &path);
bool file_exists(const std::string &path);
std::string join_path(const std::string &a, const std::string &b);
void sleep_ms(int ms);
void log_tool(bool debug, const char* level, const char* fmt, ...);
bool open_camera(const FacialAuthConfig &cfg, cv::VideoCapture &cap, std::string &device_used);
bool detect_face(const FacialAuthConfig &cfg, const cv::Mat &frame, cv::Rect &face_roi,
                 cv::CascadeClassifier &haar, cv::dnn::Net &dnn);
void load_detectors(const FacialAuthConfig &cfg, cv::CascadeClassifier &haar, cv::dnn::Net &dnn,
                    bool &use_dnn, std::string &log);

#endif // LIBFACIALAUTH_H
