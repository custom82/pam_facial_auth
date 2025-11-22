#ifndef LIBFACIALAUTH_H
#define LIBFACIALAUTH_H

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <opencv2/dnn.hpp>

// ==========================================================
// Default config file path
// ==========================================================
#ifndef FACIALAUTH_CONFIG_DEFAULT
#define FACIALAUTH_CONFIG_DEFAULT "/etc/pam_facial_auth/pam_facial.conf"
#endif

// ==========================================================
// Global configuration
// ==========================================================

struct FacialAuthConfig {
    // Path to config file (informative)
    std::string config_path = FACIALAUTH_CONFIG_DEFAULT;

    // Base directory for images and models:
    //   images: <basedir>/images/<user>/
    //   models: <basedir>/models/<user>.xml
    std::string basedir = "/etc/pam_facial_auth";

    // Video device (if empty, use camera_index)
    std::string device;

    // Camera index when device is empty
    int camera_index = 0;

    // Number of frames for train/test
    int frames = 10;

    // Normalized face size (grayscale patch)
    int width  = 200;
    int height = 200;

    // Milliseconds between frames
    int sleep_ms = 150;

    // Threshold for classic methods (LBPH/Eigen/Fisher)
    double threshold = 60.0;

    // Debug / overwrite
    bool debug           = false;
    bool force_overwrite = false;

    // ======================================================
    // Generic DNN parameters
    // ======================================================

    // "caffe", "tensorflow", "onnx", "openvino", "tflite", "torch"
    std::string dnn_type = "onnx";

    // main DNN model (when no profile is used)
    std::string dnn_model_path;
    // optional config/proto (caffe prototxt, TF pbtxt, OpenVINO .bin, ...)
    std::string dnn_proto_path;

    // backend/target: "cpu", "cuda", "opencl", "openvino"
    std::string dnn_device = "cpu";

    // logical threshold [0–1] for DNN backends
    double dnn_threshold = 0.6;

    // default DNN profile for recognition (fast, sface, lresnet100, openface, ...)
    std::string dnn_profile = "fast";

    // ======================================================
    // Detector configuration (for face detection gate)
    // ======================================================

    // detector_profile:
    //   "", "yunet", "det_uint8", "det_caffe", "det_fp16"
    std::string detector_profile;

    // detection score threshold [0–1] (for SSD-like detectors)
    double detector_threshold = 0.6;

    // absolute path to Haar cascade; if empty use defaults
    std::string haar_cascade;

    // ======================================================
    // Per-profile model paths (all absolute)
    // ======================================================

    // 1) Face recognition / embedding
    std::string dnn_model_fast;          // face_recognizer_fast.onnx
    std::string dnn_model_sface;         // face_recognition_sface_2021dec.onnx
    std::string dnn_model_lresnet100;    // LResNet100E_IR.onnx
    std::string dnn_model_openface;      // openface_nn4.small2.v1.t7

    // 2) Face detectors (SSD / YuNet)
    std::string dnn_model_yunet;               // yunet-202303.onnx
    std::string dnn_model_detector_caffe;      // opencv_face_detector.caffemodel
    std::string dnn_model_detector_fp16;       // opencv_face_detector_fp16.caffemodel
    std::string dnn_model_detector_uint8;      // opencv_face_detector_uint8.pb
    std::string dnn_proto_detector_caffe;      // deploy.prototxt for caffe detectors

    // 3) Emotion / keypoints / MediaPipe TFLite
    std::string dnn_model_emotion;                 // emotion_ferplus.onnx
    std::string dnn_model_keypoints;               // facial_keypoints.onnx
    std::string dnn_model_face_landmark_tflite;    // face_landmark.tflite
    std::string dnn_model_face_detection_tflite;   // face_detection_short_range.tflite
    std::string dnn_model_face_blendshapes_tflite; // face_blendshapes.tflite
};

// ==========================================================
// Config API
// ==========================================================

// Load key=value configuration from a file path.
// Returns false on hard error (file missing, unreadable, ...).
bool fa_load_config(const std::string &path,
                    FacialAuthConfig &cfg,
                    std::string &log);

// Select a DNN *recognition* profile and update
//   cfg.dnn_type / dnn_model_path / dnn_proto_path
// profile can be: fast, sface, lresnet100, openface,
//                 yunet, emotion, keypoints,
//                 det_uint8, det_caffe, det_fp16,
//                 mp_landmark, mp_face, mp_blend
bool fa_select_dnn_profile(FacialAuthConfig &cfg,
                           const std::string &profile,
                           std::string &log);

// User-specific paths
std::string fa_user_image_dir(const FacialAuthConfig &cfg,
                              const std::string &user);

std::string fa_user_model_path(const FacialAuthConfig &cfg,
                               const std::string &user);

// ==========================================================
// Wrapper
// ==========================================================

class FaceRecWrapper {
public:
    FaceRecWrapper();
    explicit FaceRecWrapper(const std::string &modelType_);

    bool Load(const std::string &modelFile);
    bool Save(const std::string &modelFile) const;

    bool Train(const std::vector<cv::Mat> &images,
               const std::vector<int> &labels);

    bool Predict(const cv::Mat &faceGray,
                 int &label,
                 double &confidence);

    // Detect a face ROI in a BGR or GRAY frame.
    // Returns true and fills faceROI on success.
    bool DetectFace(const cv::Mat &frame, cv::Rect &faceROI);

    // Configure DNN recognition backend (sface/fast/LResNet/OpenFace)
    void ConfigureDNN(const FacialAuthConfig &cfg);

    // Configure detector (DNN detector + Haar fallback)
    void ConfigureDetector(const FacialAuthConfig &cfg);

    bool IsDNN() const { return use_dnn; }
    double GetDnnThreshold() const { return dnn_threshold; }

private:
    std::string modelType;  // "lbph", "eigen", "fisher", "dnn"
    cv::Ptr<cv::face::FaceRecognizer> recognizer;

    // ------ DNN recognition state ------
    bool        use_dnn        = false;
    bool        dnn_loaded     = false;

    std::string dnn_profile;    // fast, sface, lresnet100, ...
    std::string dnn_type;
    std::string dnn_model_path;
    std::string dnn_proto_path;
    std::string dnn_device;
    double      dnn_threshold  = 0.6;
    cv::dnn::Net dnn_net;

    // single-user template embedding (mean embedding)
    cv::Mat dnn_template;
    bool    has_dnn_template = false;

    // ------ Detector state ------
    bool        use_dnn_detector   = false;
    std::string detector_profile;
    double      detector_threshold = 0.6;
    cv::dnn::Net dnn_detector_net;

    std::string haar_cascade_path;
    mutable cv::CascadeClassifier faceCascade;

    // ------ helpers ------
    bool load_dnn_from_model_file(const std::string &modelFile);
    bool compute_dnn_embedding(const cv::Mat &faceGray, cv::Mat &embedding);
    bool predict_with_dnn(const cv::Mat &faceGray,
                          int &label,
                          double &confidence);
    bool dnn_detector_accepts(const cv::Mat &frame, cv::Rect &faceROI);
};

// ==========================================================
// High-level API
// ==========================================================

// Capture images from webcam and save to basedir/images/<user>/img_XXX.<fmt>.
// img_format: "png", "jpg", ...
bool fa_capture_images(const std::string &user,
                       const FacialAuthConfig &cfg,
                       bool force,
                       std::string &log,
                       const std::string &img_format = "png");

// Train a model for the user (LBPH / Eigen / Fisher / DNN)
// - method: "lbph", "eigen", "fisher", "dnn"
// - inputDir: if empty uses fa_user_image_dir(cfg, user)
// - outputModel: if empty uses fa_user_model_path(cfg, user)
bool fa_train_user(const std::string &user,
                   const FacialAuthConfig &cfg,
                   const std::string &method,
                   const std::string &inputDir,
                   const std::string &outputModel,
                   bool force,
                   std::string &log);

// Test the user in realtime (camera) using the model XML
// - modelPath: if empty uses fa_user_model_path(cfg, user)
// - best_conf: best (lowest) confidence found
// - best_label: associated label
bool fa_test_user(const std::string &user,
                  const FacialAuthConfig &cfg,
                  const std::string &modelPath,
                  double &best_conf,
                  int &best_label,
                  std::string &log);

// ==========================================================
// Maintenance helpers and root check
// ==========================================================
bool fa_clean_images(const FacialAuthConfig &cfg, const std::string &user);
bool fa_clean_model(const FacialAuthConfig &cfg, const std::string &user);
void fa_list_images(const FacialAuthConfig &cfg, const std::string &user);
bool fa_check_root(const char *tool_name);

// ==========================================================
// CLI helpers (used by facial_capture / facial_training / facial_test)
// ==========================================================

int fa_training_cli(int argc, char *argv[]);
int fa_capture_cli(int argc, char *argv[]);
int fa_test_cli(int argc, char *argv[]);

#endif // LIBFACIALAUTH_H
