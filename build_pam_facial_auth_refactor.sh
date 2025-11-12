#!/usr/bin/env bash
set -e

PKG_NAME="pam_facial_auth-refactor-v2"
ROOT_DIR="$PWD/$PKG_NAME"
SRC_DIR="$ROOT_DIR/src"
MODEL_DIR="$ROOT_DIR/models"
CONF_DIR="$ROOT_DIR/config"

echo "=== Building $PKG_NAME structure ==="

# Clean up previous
rm -rf "$ROOT_DIR" "$PKG_NAME.tar.gz"
mkdir -p "$SRC_DIR" "$MODEL_DIR" "$CONF_DIR"

# -------------------------------
# 1. Scarica modelli DNN ufficiali OpenCV
# -------------------------------
echo "[+] Downloading OpenCV DNN models..."
wget -q --show-progress -O "$MODEL_DIR/deploy.prototxt" \
  https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt

wget -q --show-progress -O "$MODEL_DIR/res10_300x300_ssd_iter_140000_fp16.caffemodel" \
  https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000_fp16.caffemodel

# -------------------------------
# 2. Crea struttura sorgente C++
# -------------------------------
echo "[+] Creating refactored source tree..."

# Libreria comune
cat > "$SRC_DIR/FacialAuthCore.h" <<'EOF'
#ifndef FACIAL_AUTH_CORE_H
#define FACIAL_AUTH_CORE_H

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <string>
#include <syslog.h>

struct FacialAuthConfig {
    std::string device = "/dev/video0";
    int width = 640;
    int height = 480;
    int threshold = 75;
    int timeout = 10;
    bool nogui = true;
    bool debug = true;
    std::string model = "lbph";
};

void log_msg(const std::string &msg, bool debug = true);
bool read_config(FacialAuthConfig &cfg);
cv::Ptr<cv::face::FaceRecognizer> create_model(const std::string &model);

#endif
EOF

cat > "$SRC_DIR/FacialAuthCore.cpp" <<'EOF'
#include "FacialAuthCore.h"
#include <fstream>
#include <sstream>

void log_msg(const std::string &msg, bool debug) {
    if (debug) syslog(LOG_INFO, "%s", msg.c_str());
}

bool read_config(FacialAuthConfig &cfg) {
    std::ifstream f("/etc/pam_facial_auth/pam_facial.conf");
    if (!f.is_open()) return false;
    std::string key, value;
    while (f >> key >> value) {
        if (key == "device") cfg.device = value;
        else if (key == "width") cfg.width = std::stoi(value);
        else if (key == "height") cfg.height = std::stoi(value);
        else if (key == "threshold") cfg.threshold = std::stoi(value);
        else if (key == "timeout") cfg.timeout = std::stoi(value);
        else if (key == "nogui") cfg.nogui = (value == "true");
        else if (key == "debug") cfg.debug = (value == "true");
        else if (key == "model") cfg.model = value;
    }
    return true;
}

cv::Ptr<cv::face::FaceRecognizer> create_model(const std::string &model) {
    if (model == "eigen")
        return cv::face::EigenFaceRecognizer::create();
    else if (model == "fisher")
        return cv::face::FisherFaceRecognizer::create();
    else
        return cv::face::LBPHFaceRecognizer::create();
}
EOF

# PAM module
cat > "$SRC_DIR/pam_facial_auth.cpp" <<'EOF'
#include "FacialAuthCore.h"
#include <security/pam_modules.h>
#include <security/pam_ext.h>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

extern "C" {

PAM_EXTERN int pam_sm_authenticate(pam_handle_t *pamh, int flags, int argc, const char **argv) {
    FacialAuthConfig cfg;
    read_config(cfg);
    log_msg("pam_facial_auth: Starting authentication...", cfg.debug);

    const char *user = nullptr;
    pam_get_user(pamh, &user, nullptr);
    if (!user) {
        pam_syslog(pamh, LOG_ERR, "No username available");
        return PAM_AUTH_ERR;
    }
    log_msg(std::string("pam_facial_auth: Authenticating user ") + user, cfg.debug);

    cv::VideoCapture cap(cfg.device);
    if (!cap.isOpened()) {
        pam_syslog(pamh, LOG_ERR, "Unable to open webcam %s", cfg.device.c_str());
        return PAM_AUTH_ERR;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, cfg.width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.height);

    auto recognizer = create_model(cfg.model);
    recognizer->read(std::string("/var/lib/pam_facial_auth/") + user + ".yml");

    cv::Mat frame, gray;
    int prediction = -1;
    double confidence = 0.0;

    auto start = std::chrono::steady_clock::now();
    while (true) {
        cap >> frame;
        if (frame.empty()) continue;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        recognizer->predict(gray, prediction, confidence);
        if (cfg.debug) syslog(LOG_INFO, "Prediction=%d confidence=%.2f", prediction, confidence);

        if (confidence < cfg.threshold) {
            pam_syslog(pamh, LOG_INFO, "Facial auth success for %s", user);
            return PAM_SUCCESS;
        }

        if (std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - start).count() > cfg.timeout)
            break;
    }

    pam_syslog(pamh, LOG_ERR, "Facial auth failed for %s", user);
    return PAM_AUTH_ERR;
}

PAM_EXTERN int pam_sm_setcred(pam_handle_t *, int, int, const char **) {
    return PAM_SUCCESS;
}

}
EOF

# -------------------------------
# 3. File di configurazione base
# -------------------------------
cat > "$CONF_DIR/pam_facial.conf" <<'EOF'
device /dev/video0
width 640
height 480
threshold 75
timeout 10
nogui true
debug true
model lbph
EOF

# -------------------------------
# 4. CMakeLists
# -------------------------------
cat > "$ROOT_DIR/CMakeLists.txt" <<'EOF'
cmake_minimum_required(VERSION 3.10)
project(pam_facial_auth_refactor)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} src)

add_library(facialauth SHARED src/FacialAuthCore.cpp)
target_link_libraries(facialauth ${OpenCV_LIBS})

add_library(pam_facial_auth MODULE src/pam_facial_auth.cpp)
target_link_libraries(pam_facial_auth facialauth ${OpenCV_LIBS})
set_target_properties(pam_facial_auth PROPERTIES PREFIX "" SUFFIX ".so")

install(TARGETS pam_facial_auth LIBRARY DESTINATION /lib/security)
install(FILES config/pam_facial.conf DESTINATION /etc/pam_facial_auth)
install(FILES models/deploy.prototxt models/res10_300x300_ssd_iter_140000_fp16.caffemodel DESTINATION /usr/share/pam_facial_auth/models)
EOF

# -------------------------------
# 5. Crea archivio finale
# -------------------------------
echo "[+] Creating tarball..."
tar -czf "$PKG_NAME.tar.gz" -C "$ROOT_DIR/.." "$PKG_NAME"

echo "✅ Done! Created: $PKG_NAME.tar.gz"
ls -lh "$PKG_NAME.tar.gz"
