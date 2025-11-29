#ifndef LIBFACIALAUTH_H
#define LIBFACIALAUTH_H

// =============================================================
// libfacialauth.h - Public API for pam_facial_auth library
// OpenCV 4.12 + classic recognizers + SFace (ONNX via DNN)
// =============================================================

#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/face.hpp>

// Percorso di default del file di configurazione globale
#define FACIALAUTH_CONFIG_DEFAULT "/etc/security/pam_facial.conf"

// =============================================================
// Config strutturata
// =============================================================

struct FacialAuthConfig {
    // Directory base di lavoro: conterrà:
    //   <basedir>/images/<user>/...   (immagini di training)
    //   <basedir>/models/<user>.xml   (modelli utente)
    std::string basedir;

    // Dispositivo video
    std::string device;
    bool        fallback_device;

    int  width;
    int  height;
    int  frames;      // quanti frame in cattura/test
    int  sleep_ms;    // pausa tra frame

    bool debug;
    bool nogui;

    // Detector
    //   "auto"        -> prova YuNet se disponibile, poi Haar
    //   "yunet"       -> YuNet FP32
    //   "yunet_int8"  -> YuNet INT8
    //   "haar"        -> solo Haar
    //   "none"        -> nessun detector (non consigliato)
    std::string detector_profile;

    // Path modelli detector
    std::string haar_cascade_path;
    std::string yunet_model;
    std::string yunet_model_int8;

    // SFace (ONNX via DNN)
    std::string sface_model;
    std::string sface_model_int8;  // INT8, più veloce
    bool        sface_prefer_fp32;

    double sface_threshold;
    double lbph_threshold;
    double eigen_threshold;
    double fisher_threshold;

    // Tipo di recognizer da usare di default:
    //   "lbph", "eigen", "fisher", "sface", "auto"
    std::string recognizer;

    // Logging
    std::string log_file;
    bool        force_overwrite;

    FacialAuthConfig();
};

// =============================================================
// Wrapper classico per LBPH/Eigen/Fisher
// =============================================================

class FaceRecWrapper {
public:
    explicit FaceRecWrapper(const std::string &type);

    bool CreateRecognizer();
    bool InitCascade(const std::string &cascadePath);
    bool Load(const std::string &file);
    bool Save(const std::string &file) const;
    bool Train(const std::vector<cv::Mat> &images,
               const std::vector<int>    &labels);
    bool Predict(const cv::Mat &face,
                 int &prediction,
                 double &confidence) const;
                 bool DetectFace(const cv::Mat &frame,
                                 cv::Rect &faceROI);

                 std::string getModelType() const { return modelType; }

private:
    std::string                      modelType;
    cv::Ptr<cv::face::FaceRecognizer> recognizer;
    cv::CascadeClassifier            faceCascade;
};

// =============================================================
// Funzioni di utilità / API
// =============================================================

// Lettura file di configurazione stile key=value
bool read_kv_config(const std::string &path,
                    FacialAuthConfig  &cfg,
                    std::string       *logbuf);

// Costruzione percorsi standard immagini / modelli
std::string fa_user_image_dir(const FacialAuthConfig &cfg,
                              const std::string      &user);
std::string fa_user_model_path(const FacialAuthConfig &cfg,
                               const std::string      &user);

// Funzioni principali di alto livello
bool fa_capture(const std::string      &user,
                const FacialAuthConfig &cfg_override,
                std::string            &logbuf);

bool fa_train(const std::string      &user,
              const FacialAuthConfig &cfg_override,
              const std::string      &method,
              std::string            &logbuf);

bool fa_test(const std::string      &user,
             const FacialAuthConfig &cfg_override,
             double                 &confidence,
             std::string            &logbuf);

// Funzioni di manutenzione
bool fa_clean_images(const FacialAuthConfig &cfg,
                     const std::string      &user);
bool fa_clean_model(const FacialAuthConfig &cfg,
                    const std::string      &user);
void fa_list_images (const FacialAuthConfig &cfg,
                     const std::string      &user);

// Controllo root per i tool CLI / API
bool fa_check_root(const char *tool_name);

// Entry point per i tool da riga di comando
int facial_capture_cli_main (int argc, char *argv[]);
int facial_training_cli_main(int argc, char *argv[]);
int facial_test_cli_main    (int argc, char *argv[]);

#endif // LIBFACIALAUTH_H
