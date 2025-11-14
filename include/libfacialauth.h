#ifndef LIBFACIALAUTH_H
#define LIBFACIALAUTH_H

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <opencv2/objdetect.hpp>
#include <string>
#include <vector>

// ==========================================================
// Configurazione globale
// ==========================================================
struct FacialAuthConfig {
    bool debug           = false;
    bool nogui           = false;
    double threshold     = 80.0;                 // soglia per LBPH (più bassa = match migliore)
    int    timeout       = 10;                   // secondi per PAM/test
    std::string basedir  = "/etc/pam_facial_auth";
    std::string device   = "/dev/video0";
    int width            = 640;
    int height           = 480;
    int frames           = 5;                    // quanti frame provare
    bool fallback_device = true;                 // prova /dev/video1 se /dev/video0 fallisce
    int sleep_ms         = 500;                  // pausa tra catture (ms)
    std::string model_path;                      // Model path (added)
};

// ==========================================================
// Utility di base (config, path, logging, camera)
// ==========================================================
std::string trim(const std::string &s);
bool str_to_bool(const std::string &s, bool defval);
bool read_kv_config(const std::string &path, FacialAuthConfig &cfg, std::string *logbuf = nullptr);
void ensure_dirs(const std::string &path);
bool file_exists(const std::string &path);
std::string join_path(const std::string &a, const std::string &b);
void sleep_ms(int ms);
void log_tool(bool debug, const char* level, const char* fmt, ...);
bool open_camera(const FacialAuthConfig &cfg, cv::VideoCapture &cap, std::string &device_used);

// Path helper
std::string fa_user_image_dir(const FacialAuthConfig &cfg, const std::string &user);
std::string fa_user_model_path(const FacialAuthConfig &cfg, const std::string &user);

// ==========================================================
// Wrapper per riconoscitore facciale
// ==========================================================
class FaceRecWrapper {
public:
    explicit FaceRecWrapper(const std::string &modelType = "LBPH");

    bool Load(const std::string &modelFile);
    bool Save(const std::string &modelFile) const;
    bool Train(const std::vector<cv::Mat> &images, const std::vector<int> &labels);
    bool Predict(const cv::Mat &face, int &prediction, double &confidence) const;

    // Rilevamento volto con Haar cascade
    bool DetectFace(const cv::Mat &frame, cv::Rect &faceROI);

private:
    cv::Ptr<cv::face::LBPHFaceRecognizer> recognizer;
    cv::CascadeClassifier faceCascade;
    std::string modelType;
};

// ==========================================================
// API di alto livello usata da binari & modulo PAM
// ==========================================================

// Cattura immagini del volto per l'utente e le salva in basedir/images/<user>
bool fa_capture_images(const std::string &user,
                       const FacialAuthConfig &cfg,
                       bool force,
                       std::string &log);

// Addestra il modello per l'utente a partire da una directory di training
bool fa_train_user(const std::string &user,
                   const FacialAuthConfig &cfg,
                   const std::string &method,        // solo "lbph" supportato al momento
                   const std::string &inputDir,      // directory immagini (es. basedir/images/user)
const std::string &outputModel,   // path file XML da salvare
bool force,
std::string &log);

// Testa il volto dell'utente con la webcam usando il modello dato
// Ritorna true se il match supera la soglia; best_conf & best_label sono riempiti
bool fa_test_user(const std::string &user,
                  const FacialAuthConfig &cfg,
                  const std::string &modelPath,
                  double &best_conf,
                  int &best_label,
                  std::string &log);

#endif // LIBFACIALAUTH_H
