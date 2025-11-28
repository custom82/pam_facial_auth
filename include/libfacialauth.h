#ifndef LIBFACIALAUTH_H
#define LIBFACIALAUTH_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

// ==========================================================
// CONFIG STRUCTURE
// ==========================================================

struct FacialAuthConfig {

    // Directory base
    std::string basedir = "/etc/pam_facial_auth";

    // Video
    std::string device = "/dev/video0";
    int width  = 640;
    int height = 480;

    int frames   = 15;
    int sleep_ms = 200;

    bool nogui  = true;
    bool debug  = false;
    bool fallback_device = false;

    // Thresholds for classical recognizers
    double lbph_threshold   = 80.0;
    double eigen_threshold  = 5000.0;
    double fisher_threshold = 500.0;

    // PCA components
    int eigen_components  = 100;
    int fisher_components = 10;

    // Force override (not used in CLI now)
    bool force_overwrite = false;
    bool ignore_failure  = false;

    std::string model_path;
    std::string haar_cascade_path;

    // ------------------------------
    // Modern DNN pipeline (YUNet + SFace)
    // ------------------------------

    // Face detector: "haar", "yunet_cpu", "yunet_cuda"
    std::string detector_profile = "haar";

    // Path ai modelli DNN
    std::string yunet_model;   // ONNX YUNet
    std::string sface_model;   // ONNX SFace recognizer

    // SFace cosine distance threshold
    double sface_threshold = 0.50;

    // Directory embedding (default → basedir/embeddings)
    std::string embeddings_dir;

    // Preferred training method: lbph/eigen/fisher/sface
    std::string training_method;

    // Logfile opzionale
    std::string log_file;
};

#define FACIALAUTH_CONFIG_DEFAULT "/etc/security/pam_facial.conf"


// ==========================================================
// Utility functions
// ==========================================================

std::string trim(const std::string &s);
bool        str_to_bool(const std::string &s, bool defval);

bool read_kv_config(const std::string &path,
                    FacialAuthConfig &cfg,
                    std::string *logbuf = nullptr);

bool file_exists(const std::string &path);

// Path helpers
std::string fa_user_image_dir(const FacialAuthConfig &cfg,
                              const std::string &user);

std::string fa_user_model_path(const FacialAuthConfig &cfg,
                               const std::string &user);

std::string fa_user_embedding_path(const FacialAuthConfig &cfg,
                                   const std::string &user);

// Detect model type from XML header
std::string fa_detect_model_type(const std::string &xmlPath);


// ==========================================================
// FaceRecWrapper (LBPH / Eigen / Fisher)
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

                 bool InitCascade(const std::string &cascadePath);
                 bool CreateRecognizer();

                 const std::string &GetModelType() const { return modelType; }

private:
    std::string modelType;
    cv::Ptr<cv::face::FaceRecognizer> recognizer;
    cv::CascadeClassifier faceCascade;
};


// ==========================================================
// HIGH LEVEL API
// ==========================================================

// Capture images using HAAR or YUNet
bool fa_capture_images(const std::string &user,
                       const FacialAuthConfig &cfg,
                       bool force,
                       std::string &logbuf,
                       const std::string &img_format = "jpg");

// Training (classic or SFace embedding)
bool fa_train_user(const std::string &user,
                   const FacialAuthConfig &cfg,
                   const std::string &method,
                   const std::string &inputDir,
                   const std::string &outputModel,
                   bool force,
                   std::string &logbuf);

// Authentication
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
// CLI ENTRY POINTS
// ==========================================================

int facial_capture_cli_main (int argc, char *argv[]);
int facial_training_cli_main(int argc, char *argv[]);
int facial_test_cli_main    (int argc, char *argv[]);

#endif // LIBFACIALAUTH_H
