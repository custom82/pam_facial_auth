#ifndef LIBFACIALAUTH_H
#define LIBFACIALAUTH_H

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>  // OpenCV face module
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>
#include <syslog.h>

// Funzione di utilità per rimuovere spazi bianchi da una stringa
std::string trim(const std::string &s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

// Funzione di utilità per convertire stringhe in booleani
bool str_to_bool(const std::string &s, bool defval) {
    auto t = trim(s);
    for (auto &c : t) c = ::tolower(c);
    if (t == "1" || t == "true" || t == "yes" || t == "on") return true;
    if (t == "0" || t == "false" || t == "no"  || t == "off") return false;
    return defval;
}

// Classe per la gestione del riconoscimento facciale tramite OpenCV
class FaceRecWrapper {
public:
    FaceRecWrapper(const std::string& modelPath, const std::string& name, const std::string& model_type);
    void Train(const std::vector<cv::Mat>& images, const std::vector<int>& labels);
    void Recognize(cv::Mat& face);
    void Load(const std::string& modelFile);
    void Save(const std::string& modelFile);
    void Predict(cv::Mat& face, int& prediction, double& confidence);

private:
    cv::Ptr<cv::face::LBPHFaceRecognizer> recognizer;
    std::string modelType;
};

// Classe per la gestione dell'autenticazione facciale
class FacialAuth {
public:
    FacialAuth();
    ~FacialAuth();

    bool Authenticate(const std::string &user);  // Metodo per autenticare un utente

private:
    bool LoadModel(const std::string &modelPath); // Carica il modello di riconoscimento facciale
    bool RecognizeFace(const cv::Mat &faceImage);  // Riconosce il volto in un'immagine

    cv::Ptr<cv::face::LBPHFaceRecognizer> recognizer;  // Riconoscitore facciale OpenCV
    std::string modelPath;  // Percorso del modello facciale
};

// Funzione di logging per il tool di autenticazione facciale
void log_tool(bool debug, const char* level, const char* fmt, ...) {
    if (!debug && std::string(level) == "DEBUG") return;
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    std::cerr << "[FA-" << level << "] " << buf << std::endl;
}

// Classe principale per l'integrazione con PAM (modulo PAM facciale)
class PAMFacialAuth {
public:
    PAMFacialAuth();
    ~PAMFacialAuth();

    // Metodo di autenticazione dell'utente
    int authenticate(pam_handle_t *pamh, int flags, int argc, const char **argv);

private:
    // Metodo per ottenere l'utente dal PAM handle
    const char* getUser(pam_handle_t *pamh);

    int retval;  // Stato dell'autenticazione
};

// Funzione per leggere la configurazione da un file
bool read_kv_config(const std::string &path, FacialAuthConfig &cfg, std::string *logbuf=nullptr);

// Funzione per assicurarsi che la directory esista
void ensure_dirs(const std::string &path);

// Funzione per verificare se un file esiste
bool file_exists(const std::string &path);

// Funzione per unire due percorsi
std::string join_path(const std::string &a, const std::string &b);

// Funzione per il controllo dei volti nelle immagini
bool detect_face(const FacialAuthConfig &cfg, const cv::Mat &frame, cv::Rect &face_roi,
                 cv::CascadeClassifier &haar, cv::dnn::Net &dnn);

// Funzione per aprire la videocamera
bool open_camera(const FacialAuthConfig &cfg, cv::VideoCapture &cap, std::string &device_used);

#endif // LIBFACIALAUTH_H
