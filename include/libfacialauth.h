/*
 * Project: pam_facial_auth
 * License: GPL-3.0
 *
 * Security requirement:
 *  - Model file is always stored in: /etc/security/pam_facial_auth/<user>.xml
 *  - This avoids dependency on /var being mounted at boot.
 *
 * Model format (XML):
 *  - pfa_header { version, algorithm, ... }
 *  - For SFace: embeddings matrix (NxD CV_32F)
 *  - For classic (LBPH/Eigen/Fisher): OpenCV FaceRecognizer serialized data
 */

#ifndef LIBFACIALAUTH_H
#define LIBFACIALAUTH_H

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

struct FacialAuthConfig {
    // NOTE: basedir is still used for captures (temporary images).
    std::string basedir = "/var/lib/pam_facial_auth";

    // Capture device
    std::string device = "/dev/video0";
    int width = 640;
    int height = 480;

    // Detector selection
    std::string detector = "none";   // none, cascade, yunet
    std::string cascade_path;        // Haar cascade XML (if detector=cascade)
    std::string detect_yunet;        // YuNet ONNX (if detector=yunet)

    // Recognizer selection
    std::string method = "auto";     // auto, lbph, eigen, fisher, sface
    std::string recognize_sface;     // SFace ONNX (if method=sface)

    // Capture parameters
    std::string image_format = "jpg";
    int frames = 30;                // number of frames to capture / try
    int sleep_ms = 100;             // delay between frames in ms (for capture/test)

    // Thresholds
    // For classic recognizers: smaller confidence is better.
    double threshold = 60.0;        // generic classic threshold fallback
    double lbph_threshold = 60.0;
    double eigen_threshold = 5000.0;
    double fisher_threshold = 500.0;

    // For SFace: larger similarity is better (cosine similarity).
    double sface_threshold = 0.36;

    // Output / debug
    bool debug = false;
    bool verbose = false;
    bool nogui = true;

    // PAM behavior
    bool ignore_failure = false;
};

// Base interface for recognizer plugins
class RecognizerPlugin {
public:
    virtual ~RecognizerPlugin() = default;

    virtual bool load(const std::string& path, std::string& err) = 0;
    virtual bool train(const std::vector<cv::Mat>& faces,
                       const std::vector<int>& labels,
                       const std::string& save_path,
                       std::string& err) = 0;
                       virtual bool predict(const cv::Mat& face, int& label, double& confidence, std::string& err) = 0;

                       // Return true if confidence indicates a match, based on plugin semantics + config thresholds.
                       virtual bool is_match(double confidence, const FacialAuthConfig& cfg) const = 0;

                       virtual std::string get_name() const = 0; // lbph/eigen/fisher/sface
};

#ifdef __cplusplus
extern "C" {
    #endif

    bool fa_check_root(const std::string& tool_name);

    // Load configuration from file (default: /etc/pam_facial_auth/pam_facial.conf)
    bool fa_load_config(FacialAuthConfig& cfg, std::string& log, const std::string& path = "");

    // Always returns: /etc/security/pam_facial_auth/<user>.xml
    std::string fa_user_model_path(const FacialAuthConfig& cfg, const std::string& user);

    // Capture/training/testing
    bool fa_clean_captures(const std::string& user, const FacialAuthConfig& cfg, std::string& log);
    bool fa_capture_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& device_path, std::string& log);
    bool fa_train_user(const std::string& user, const FacialAuthConfig& cfg, std::string& log);

    // Returns true if authenticated (match), false otherwise. Still fills confidence/label/log.
    bool fa_test_user(const std::string& user,
                      const FacialAuthConfig& cfg,
                      const std::string& model_path,
                      double& confidence,
                      int& label,
                      std::string& log);

    #ifdef __cplusplus
}
#endif

#endif // LIBFACIALAUTH_H
