#ifndef LIBFACIALAUTH_H
#define LIBFACIALAUTH_H

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>  // OpenCV face module

// Definizione della classe FaceRecWrapper
class FaceRecWrapper {
public:
    FaceRecWrapper(const std::string& modelPath, const std::string& name, const std::string& model_type);
    void Train(const std::vector<cv::Mat>& images, const std::vector<int>& labels);
    void Recognize(cv::Mat& face);
    void Load(const std::string& modelFile);
    void Save(const std::string& modelFile);  // Nuovo metodo per salvare il modello
    void Predict(cv::Mat& face, int& prediction, double& confidence);

private:
    cv::Ptr<cv::face::LBPHFaceRecognizer> recognizer;
    std::string modelType;
};

// Definizione della classe FacialAuth
class FacialAuth {
public:
    FacialAuth();
    ~FacialAuth();

    bool Authenticate(const std::string &user);  // Metodo per autenticare un utente

private:
    bool LoadModel(const std::string &modelPath); // Carica il modello di riconoscimento facciale
    bool TrainModel(const std::vector<cv::Mat> &images, const std::vector<int> &labels);  // Metodo per addestrare il modello (opzionale)
    bool RecognizeFace(const cv::Mat &faceImage);  // Riconosce il volto in un'immagine

    cv::Ptr<cv::face::LBPHFaceRecognizer> recognizer;  // Riconoscitore facciale OpenCV
    std::string modelPath;  // Percorso del modello facciale
};

// Struttura di configurazione per l'autenticazione facciale
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

// Funzioni ausiliarie
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
