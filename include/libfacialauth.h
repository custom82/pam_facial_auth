#ifndef LIBFACIALAUTH_H
#define LIBFACIALAUTH_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

// ==========================================================
// CONFIG
// ==========================================================

struct FacialAuthConfig {

    // basedir viene letta SOLO dal file pam_facial.conf
    std::string basedir = "/etc/pam_facial_auth";

    std::string device  = "/dev/video0";
    int         width   = 640;
    int         height  = 480;

    // Threshold LBPH (vecchio threshold generale diventa LBPH)
    double      threshold = 80.0;     // compatibilità retroattiva
    double      lbph_threshold   = 80.0;

    // Threshold nuovi
    double      eigen_threshold  = 3500.0;
    double      fisher_threshold = 500.0;

    int         frames    = 15;
    int         sleep_ms  = 200;

    bool        nogui    = true;
    bool        debug    = false;
    bool        fallback_device = false;

    // File modello XML
    std::string model_path;

    // Path assoluto HAAR CASCADE (obbligatorio)
    std::string haar_cascade_path;

    // metodo training (usato solo durante il training)
    std::string training_method;

    // log
    std::string log_file;

    bool force_overwrite = false;
    bool ignore_failure  = false;

    // PARAMETRI AGGIUNTIVI PER EIGEN/FISHER
    int eigen_components  = 80;     // numero componenti PCA
    int fisher_components = 60;     // numero componenti LDA
};

#define FACIALAUTH_CONFIG_DEFAULT "/etc/security/pam_facial.conf"


// ==========================================================
// UTILS
// ==========================================================

std::string trim(const std::string &s);
bool str_to_bool(const std::string &s, bool defval);

bool read_kv_config(const std::string &path,
                    FacialAuthConfig &cfg,
                    std::string *logbuf = nullptr);

bool file_exists(const std::string &path);

// Helpers per path
std::string fa_user_model_path(const FacialAuthConfig &cfg,
                               const std::string &user);

std::string fa_user_image_dir(const FacialAuthConfig &cfg,
                              const std::string &user);

// Leggi il tipo di modello dal file XML
std::string fa_detect_model_type(const std::string &xmlPath);


// ==========================================================
// FaceRecWrapper
// ==========================================================

class FaceRecWrapper {
public:
    explicit FaceRecWrapper(const std::string &modelType = "lbph");

    bool Load(const std::string &file);
    bool Save(const std::string &file) const;

    bool Train(const std::vector<cv::Mat> &images,
               const std::vector<int>    &labels);

    bool Predict(const cv::Mat &face,
                 int &prediction,
                 double &confidence) const;

    bool DetectFace(const cv::Mat &frame,
                    cv::Rect &faceROI);

    // Inizializza Haar cascade
    bool InitCascade(const std::string &cascadePath);

    // Crea LBPH/EIGEN/FISHER recognizer
    bool CreateRecognizer();

    const std::string &GetModelType() const { return modelType; }

private:
    std::string modelType;

    cv::Ptr<cv::face::FaceRecognizer> recognizer;
    cv::CascadeClassifier faceCascade;
};


// ==========================================================
// API HIGH LEVEL
// ==========================================================

bool fa_capture_images(const std::string &user,
                       const FacialAuthConfig &cfg,
                       bool force,
                       std::string &logbuf,
                       const std::string &img_format = "jpg");

bool fa_train_user(const std::string &user,
                   const FacialAuthConfig &cfg,
                   const std::string &method,
                   const std::string &inputDir,
                   const std::string &outputModel,
                   bool force,
                   std::string &logbuf);

bool fa_test_user(const std::string &user,
                  const FacialAuthConfig &cfg,
                  const std::string &modelPath,
                  double &best_conf,
                  int &best_label,
                  std::string &logbuf);

bool fa_clean_images(const FacialAuthConfig &cfg, const std::string &user);
bool fa_clean_model (const FacialAuthConfig &cfg, const std::string &user);
void fa_list_images(const FacialAuthConfig &cfg, const std::string &user);

bool fa_check_root(const char *tool_name);

// CLI wrapper
int facial_capture_cli_main(int argc, char *argv[]);
int facial_training_cli_main(int argc, char *argv[]);
int facial_test_cli_main(int argc, char *argv[]);

#endif
