#ifndef LIBFACIALAUTH_H
#define LIBFACIALAUTH_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

// ==========================================================
// Config structure
// ==========================================================
struct FacialAuthConfig {
    std::string basedir       = "/etc/pam_facial_auth";
    std::string device        = "/dev/video0";
    int         width         = 640;
    int         height        = 480;
    double      threshold     = 75.0;
    int         timeout       = 5;
    bool        nogui         = true;
    bool        debug         = false;
    int         frames        = 15;
    bool        fallback_device = false;
    int         sleep_ms      = 200;

    std::string model_path;
    std::string haar_cascade_path;
    std::string training_method;
    std::string log_file;

    bool force_overwrite      = false;
    std::string face_detection_method = "haar";
    bool ignore_failure       = false;
};

#define FACIALAUTH_CONFIG_DEFAULT "/etc/security/pam_facial.conf"

// ==========================================================
// Extra utilities used by PAM module
// ==========================================================

std::string trim(const std::string &s);
bool str_to_bool(const std::string &s, bool defval);

bool read_kv_config(const std::string &path,
                    FacialAuthConfig &cfg,
                    std::string *logbuf = nullptr);

// Path helpers
std::string fa_user_model_path(const FacialAuthConfig &cfg, const std::string &user);
std::string fa_user_image_dir(const FacialAuthConfig &cfg, const std::string &user);

// ==========================================================
// FaceRecWrapper class
// ==========================================================
class FaceRecWrapper {
public:
    explicit FaceRecWrapper(const std::string &modelType = "lbph");

    bool Load(const std::string &file);
    bool Save(const std::string &file) const;
    bool Train(const std::vector<cv::Mat> &images,
               const std::vector<int> &labels);

    bool Predict(const cv::Mat &face,
                 int &prediction,
                 double &confidence) const;

                 bool DetectFace(const cv::Mat &frame, cv::Rect &faceROI);

private:
    std::string modelType;
    cv::Ptr<cv::face::LBPHFaceRecognizer> recognizer;
    cv::CascadeClassifier faceCascade;
};

// ==========================================================
// API
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
bool fa_clean_model(const FacialAuthConfig &cfg, const std::string &user);
void fa_list_images(const FacialAuthConfig &cfg, const std::string &user);
bool fa_check_root(const char *tool_name);

// CLI wrappers
int facial_capture_cli_main(int argc, char *argv[]);
int facial_training_cli_main(int argc, char *argv[]);
int facial_test_cli_main(int argc, char *argv[]);

#endif
