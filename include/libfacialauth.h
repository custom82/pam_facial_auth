#ifndef LIBFACIALAUTH_H
#define LIBFACIALAUTH_H

#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/face.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#define FACIALAUTH_CONFIG_DEFAULT "/etc/security/pam_facial.conf"

// =============================================
// Config
// =============================================

struct FacialAuthConfig {
    std::string config_path;

    std::string basedir = "/etc/pam_facial_auth";

    std::string device;   // /dev/video0 (optional)
    int camera_index = 0;

    int frames = 10;
    int width  = 200;
    int height = 200;

    int sleep_ms = 150;

    double threshold = 60.0;

    bool debug = false;
    bool force_overwrite = false;

    // ---- DNN recognizer ----
    std::string dnn_type;
    std::string dnn_model_path;
    std::string dnn_proto_path;
    std::string dnn_device = "cpu";
    double      dnn_threshold = 0.6;
    std::string dnn_profile;

    // ---- DNN embedding models ----
    std::string dnn_model_fast;
    std::string dnn_model_sface;
    std::string dnn_model_lresnet100;
    std::string dnn_model_openface;

    // ---- DNN detectors ----
    std::string dnn_model_yunet;
    std::string dnn_model_detector_caffe;
    std::string dnn_model_detector_fp16;
    std::string dnn_model_detector_uint8;
    std::string dnn_proto_detector_caffe;

    // ---- Extra models ----
    std::string dnn_model_emotion;
    std::string dnn_model_keypoints;
    std::string dnn_model_face_landmark_tflite;
    std::string dnn_model_face_detection_tflite;
    std::string dnn_model_face_blendshapes_tflite;

    // ---- Haar fallback ----
    std::string haar_cascade;

    // ---- Detector selection ----
    std::string detector_profile;
    double      detector_threshold = 0.6;
    int         detector_width  = 0;
    int         detector_height = 0;
};

// =============================================
// Utility prototypes
// =============================================

bool fa_load_config(const std::string &path,
                    FacialAuthConfig &cfg,
                    std::string &log);

std::string fa_user_image_dir(const FacialAuthConfig &cfg,
                              const std::string &user);

std::string fa_user_model_path(const FacialAuthConfig &cfg,
                               const std::string &user);

// =============================================
// open_camera() — MUST BE DECLARED!
// =============================================
bool open_camera(const FacialAuthConfig &cfg,
                 cv::VideoCapture &cap,
                 std::string &device_used);

// =============================================
// FaceRecWrapper
// =============================================

class FaceRecWrapper {

public:
    FaceRecWrapper();
    explicit FaceRecWrapper(const std::string &modelType);

    bool Load(const std::string &modelFile);
    bool Save(const std::string &modelFile) const;

    bool Train(const std::vector<cv::Mat> &images,
               const std::vector<int> &labels);

    bool Predict(const cv::Mat &faceGray,
                 int &label,
                 double &confidence);

    bool DetectFace(const cv::Mat &frame, cv::Rect &roi);

    void ConfigureDNN(const FacialAuthConfig &cfg);
    void ConfigureDetector(const FacialAuthConfig &cfg);

    bool IsDNN() const { return use_dnn; }
    double GetDnnThreshold() const { return dnn_threshold; }

private:
    // Recognizer
    cv::Ptr<cv::face::FaceRecognizer> recognizer;
    std::string modelType;

    // DNN recognition
    bool        use_dnn = false;
    bool        dnn_loaded = false;
    std::string dnn_profile;
    std::string dnn_type;
    std::string dnn_model_path;
    std::string dnn_proto_path;
    std::string dnn_device;
    double      dnn_threshold = 0.6;

    cv::dnn::Net dnn_net;

    cv::Mat dnn_template;
    bool    has_dnn_template = false;

    bool load_dnn_from_model_file(const std::string &modelFile);
    bool compute_dnn_embedding(const cv::Mat &faceGray, cv::Mat &embedding);
    bool predict_with_dnn(const cv::Mat &faceGray,
                          int &label,
                          double &confidence);

    // Haar fallback
    std::string haar_path;
    mutable cv::CascadeClassifier faceCascade;

    // DNN Detector
    bool        use_dnn_detector = false;
    bool        detector_loaded  = false;

    std::string detector_profile;
    std::string detector_type;
    std::string detector_model_path;
    std::string detector_proto_path;
    std::string detector_device;
    double      detector_threshold = 0.6;
    int         detector_input_width  = 0;
    int         detector_input_height = 0;

    cv::dnn::Net detector_net;
};

// =============================================
// High level API
// =============================================

bool fa_capture_images(const std::string &user,
                       const FacialAuthConfig &cfg,
                       bool force,
                       std::string &log,
                       const std::string &format);

bool fa_train_user(const std::string &user,
                   const FacialAuthConfig &cfg,
                   const std::string &method,
                   const std::string &inputDir,
                   const std::string &outputModel,
                   bool force,
                   std::string &log);

bool fa_test_user(const std::string &user,
                  const FacialAuthConfig &cfg,
                  const std::string &modelPath,
                  double &best_conf,
                  int &best_label,
                  std::string &log);

bool fa_clean_images(const FacialAuthConfig &cfg, const std::string &user);
bool fa_clean_model (const FacialAuthConfig &cfg, const std::string &user);
void fa_list_images (const FacialAuthConfig &cfg, const std::string &user);

bool fa_check_root(const char *tool);

// =============================================
// CLI frontends
// =============================================

int fa_training_cli(int argc, char *argv[]);
int fa_capture_cli (int argc, char *argv[]);
int fa_test_cli    (int argc, char *argv[]);

#endif
