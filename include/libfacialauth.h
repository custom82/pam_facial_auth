#ifndef LIBFACIALAUTH_H
#define LIBFACIALAUTH_H

#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/face.hpp>

#define FACIALAUTH_CONFIG_DEFAULT "/etc/security/pam_facial.conf"

// ==============================================================
// Config structure
// ==============================================================

struct FacialAuthConfig {
    std::string config_path;

    // Base directory for images + models
    std::string basedir = "/etc/pam_facial_auth";

    // Camera settings
    std::string device;         // /dev/videoX (empty = use camera_index)
    int camera_index = 0;
    int frames       = 10;
    int width        = 200;
    int height       = 200;
    int sleep_ms     = 150;

    // Classical recognizer threshold
    double threshold = 60.0;

    // Flags
    bool debug = false;
    bool force_overwrite = false;

    // DNN generic
    std::string dnn_type;         // caffe | tensorflow | onnx | tflite | torch | openvino
    std::string dnn_model_path;
    std::string dnn_proto_path;
    std::string dnn_device = "cpu";
    double      dnn_threshold = 0.6;
    std::string dnn_profile;      // fast | sface | lresnet100 | openface | yunet | ...

    // DNN model groups (from pam_facial.conf)
    std::string dnn_model_fast;
    std::string dnn_model_sface;
    std::string dnn_model_lresnet100;
    std::string dnn_model_openface;

    std::string dnn_model_yunet;
    std::string dnn_model_detector_caffe;
    std::string dnn_model_detector_fp16;
    std::string dnn_model_detector_uint8;
    std::string dnn_proto_detector_caffe;

    std::string dnn_model_emotion;
    std::string dnn_model_keypoints;
    std::string dnn_model_face_landmark_tflite;
    std::string dnn_model_face_detection_tflite;
    std::string dnn_model_face_blendshapes_tflite;

    // Haar + DNN detector system
    std::string haar_cascade;      // absolute path or empty for auto-detect
    std::string detector_profile;  // det_uint8 | det_caffe | det_fp16 | ""
    double      detector_threshold = 0.6;
    int         detector_width  = 0;  // default 300x300 if omitted
    int         detector_height = 0;
};

// ==============================================================
// Helpers
// ==============================================================

bool fa_load_config(const std::string &path,
                    FacialAuthConfig &cfg,
                    std::string &log);

std::string fa_user_image_dir(const FacialAuthConfig &cfg,
                              const std::string &user);

std::string fa_user_model_path(const FacialAuthConfig &cfg,
                               const std::string &user);

// ==============================================================
// FaceRecWrapper
// ==============================================================

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

    bool DetectFace(const cv::Mat &frame, cv::Rect &faceROI);

    void ConfigureDNN(const FacialAuthConfig &cfg);
    void ConfigureDetector(const FacialAuthConfig &cfg);

    bool IsDNN() const;
    double GetDnnThreshold() const;

private:
    // ---- DNN recognition ----
    bool load_dnn_from_model_file(const std::string &modelFile);
    bool predict_with_dnn(const cv::Mat &faceGray,
                          int &label,
                          double &confidence);
    bool compute_dnn_embedding(const cv::Mat &faceGray,
                               cv::Mat &embedding);

    std::string modelType;
    cv::Ptr<cv::face::FaceRecognizer> recognizer;

    bool        use_dnn      = false;
    bool        dnn_loaded   = false;
    std::string dnn_profile;
    std::string dnn_type;
    std::string dnn_model_path;
    std::string dnn_proto_path;
    std::string dnn_device;
    double      dnn_threshold = 0.6;
    cv::dnn::Net dnn_net;

    cv::Mat dnn_template;
    bool    has_dnn_template = false;

    // ---- DNN detector + Haar fallback ----
    std::string haar_cascade_path;

    bool        use_dnn_detector = false;
    bool        detector_loaded  = false;
    std::string detector_profile;
    std::string detector_type;
    std::string detector_model_path;
    std::string detector_proto_path;
    std::string detector_device;
    double      detector_threshold   = 0.6;
    int         detector_input_width  = 0;
    int         detector_input_height = 0;
    cv::dnn::Net detector_net;

    mutable cv::CascadeClassifier faceCascade;
};

// ==============================================================
// High level capture/train/test API
// ==============================================================

bool fa_capture_images(const std::string &user,
                       const FacialAuthConfig &cfg,
                       bool force,
                       std::string &log,
                       const std::string &img_format);

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
bool fa_clean_model(const FacialAuthConfig &cfg, const std::string &user);
void fa_list_images(const FacialAuthConfig &cfg, const std::string &user);

bool fa_check_root(const char *tool_name);

// ==============================================================
// CLI (training, capture, test)
// ==============================================================

int fa_training_cli(int argc, char *argv[]);
int fa_capture_cli(int argc, char *argv[]);
int fa_test_cli(int argc, char *argv[]);

#endif // LIBFACIALAUTH_H
