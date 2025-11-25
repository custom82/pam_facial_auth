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

    // ATTENZIONE — basedir:
    // - default assoluto: /etc/pam_facial_auth
    // - può essere cambiata SOLO dal file di configurazione
    std::string basedir = "/etc/pam_facial_auth";

    std::string device  = "/dev/video0";
    int         width   = 640;
    int         height  = 480;

    int         frames    = 15;
    int         sleep_ms  = 200;

    bool        nogui    = true;
    bool        debug    = false;
    bool        fallback_device = false;

    // soglie per i vari metodi
    double      lbph_threshold   = 80.0;
    double      eigen_threshold  = 5000.0;
    double      fisher_threshold = 500.0;

    // numero componenti per eigen/fisher
    int         eigen_components  = 100;
    int         fisher_components = 10;

    // File modello (override opzionale)
    std::string model_path;

    // Path assoluto al file HAAR (OBBLIGATORIO — nessun fallback)
    std::string haar_cascade_path;

    // training method preferito (lbph/eigen/fisher) – opzionale
    std::string training_method;

    // log file opzionale
    std::string log_file;

    bool        force_overwrite = false;
    bool        ignore_failure  = false;
};

#define FACIALAUTH_CONFIG_DEFAULT "/etc/security/pam_facial.conf"

// ==========================================================
// UTILS
// ==========================================================

std::string trim(const std::string &s);
bool        str_to_bool(const std::string &s, bool defval);

bool read_kv_config(const std::string &path,
                    FacialAuthConfig &cfg,
                    std::string *logbuf = nullptr);

bool file_exists(const std::string &path);

// Helpers per path
std::string fa_user_model_path(const FacialAuthConfig &cfg,
                               const std::string &user);

std::string fa_user_image_dir(const FacialAuthConfig &cfg,
                              const std::string &user);

// Riconosce automaticamente il tipo modello dal file XML
// Ritorna "lbph", "eigen", "fisher", oppure "lbph" come fallback.
std::string fa_detect_model_type(const std::string &xmlPath);


// ==========================================================
// FaceRecWrapper
// ==========================================================

class FaceRecWrapper {
public:
    explicit FaceRecWrapper(const std::string &modelType = "lbph");

    // Carica un modello e imposta il tipo corretto
    bool Load(const std::string &file);

    // Salvataggio
    bool Save(const std::string &file) const;

    // Training
    bool Train(const std::vector<cv::Mat> &images,
               const std::vector<int>    &labels);

    // Predict
    bool Predict(const cv::Mat &face,
                 int &prediction,
                 double &confidence) const;

                 // Face detection
                 bool DetectFace(const cv::Mat &frame,
                                 cv::Rect &faceROI);

                 // Carica HAAR
                 bool InitCascade(const std::string &cascadePath);

                 // Crea riconoscitore corretto (LBPH/EIGEN/FISHER)
                 bool CreateRecognizer();

                 const std::string &GetModelType() const { return modelType; }

private:
    std::string modelType;  // "lbph", "eigen", "fisher"
    cv::Ptr<cv::face::FaceRecognizer> recognizer;
    cv::CascadeClassifier faceCascade;
};


// ==========================================================
// API HIGH LEVEL
// ==========================================================

// Cattura immagini
bool fa_capture_images(const std::string &user,
                       const FacialAuthConfig &cfg,
                       bool force,
                       std::string &logbuf,
                       const std::string &img_format = "jpg");

// Training
bool fa_train_user(const std::string &user,
                   const FacialAuthConfig &cfg,
                   const std::string &method,
                   const std::string &inputDir,
                   const std::string &outputModel,
                   bool force,
                   std::string &logbuf);

// Test
bool fa_test_user(const std::string &user,
                  const FacialAuthConfig &cfg,
                  const std::string &modelPath,
                  double &best_conf,
                  int &best_label,
                  std::string &logbuf,
                  double threshold_override = -1.0);

// Maintenance
bool fa_clean_images(const FacialAuthConfig &cfg, const std::string &user);
bool fa_clean_model (const FacialAuthConfig &cfg, const std::string &user);
void fa_list_images(const FacialAuthConfig &cfg, const std::string &user);

// Root check
bool fa_check_root(const char *tool_name);


// ==========================================================
// CLI WRAPPERS
// ==========================================================

int facial_capture_cli_main (int argc, char *argv[]);
int facial_training_cli_main(int argc, char *argv[]);
int facial_test_cli_main    (int argc, char *argv[]);

#endif
