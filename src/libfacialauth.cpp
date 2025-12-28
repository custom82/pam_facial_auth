/*
 * Project: pam_facial_auth
 * License: GPL-3.0
 *
 * Security requirement:
 *   Model always stored in /etc/security/pam_facial_auth/<user>.xml
 *   Directory permissions enforced (0700) and file permissions enforced (0600).
 */

#include "libfacialauth.h"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <thread>
#include <cctype>
#include <unistd.h>
#include <sys/stat.h>
#include <cerrno>
#include <cstring>

#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

#if CV_VERSION_MAJOR >= 4
#include <opencv2/objdetect/face.hpp> // FaceDetectorYN
#endif

namespace fs = std::filesystem;

// Factories implemented in plugin_*.cpp
std::unique_ptr<RecognizerPlugin> create_classic_plugin(const std::string& method, const FacialAuthConfig& cfg);
std::unique_ptr<RecognizerPlugin> create_sface_plugin(const FacialAuthConfig& cfg);

static std::string trim(std::string s) {
    auto issp = [](unsigned char c){ return std::isspace(c); };
    while (!s.empty() && issp((unsigned char)s.front())) s.erase(s.begin());
    while (!s.empty() && issp((unsigned char)s.back())) s.pop_back();
    return s;
}

static bool parse_bool(const std::string& v) {
    std::string s = v;
    for (auto& c : s) c = (char)std::tolower((unsigned char)c);
    return (s == "1" || s == "true" || s == "yes" || s == "on");
}

static std::string default_config_path() {
    return "/etc/pam_facial_auth/pam_facial.conf";
}

// HARD security path
static std::string model_base_dir() {
    return "/etc/security/pam_facial_auth";
}

static bool ensure_secure_model_dir(std::string& err) {
    const std::string dir = model_base_dir();
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) {
        err = "Impossibile creare directory modelli " + dir + ": " + ec.message();
        return false;
    }

    // Force mode 0700 (best effort)
    if (::chmod(dir.c_str(), 0700) != 0) {
        // On some systems chmod might fail due to FS perms; still report.
        err = "chmod(0700) fallito su " + dir + ": " + std::string(std::strerror(errno));
        return false;
    }

    return true;
}

static bool chmod0600(const std::string& path, std::string& err) {
    if (::chmod(path.c_str(), 0600) != 0) {
        err = "chmod(0600) fallito su " + path + ": " + std::string(std::strerror(errno));
        return false;
    }
    return true;
}

static std::string model_algorithm_from_xml(const std::string& model_path) {
    cv::FileStorage fs(model_path, cv::FileStorage::READ);
    if (!fs.isOpened()) return "";

    cv::FileNode header = fs["pfa_header"];
    if (header.empty()) return "";

    std::string alg;
    header["algorithm"] >> alg;
    return alg;
}

static std::unique_ptr<RecognizerPlugin> create_plugin_for_method(const FacialAuthConfig& cfg, const std::string& method) {
    if (method == "sface") return create_sface_plugin(cfg);
    if (method == "lbph" || method == "eigen" || method == "fisher") return create_classic_plugin(method, cfg);
    return create_classic_plugin("lbph", cfg);
}

static bool detect_one_face(const FacialAuthConfig& cfg, const cv::Mat& bgr, cv::Mat& face_out, std::string& err) {
    if (bgr.empty()) {
        err = "detect_one_face: immagine vuota";
        return false;
    }

    if (cfg.detector == "none") {
        face_out = bgr.clone();
        return true;
    }

    if (cfg.detector == "cascade") {
        if (cfg.cascade_path.empty()) {
            err = "detector=cascade ma cascade_path è vuoto";
            return false;
        }

        cv::CascadeClassifier cc;
        if (!cc.load(cfg.cascade_path)) {
            err = "Impossibile caricare cascade: " + cfg.cascade_path;
            return false;
        }

        cv::Mat gray;
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Rect> faces;
        cc.detectMultiScale(gray, faces, 1.1, 3);

        if (faces.empty()) {
            err = "Nessun volto (cascade)";
            return false;
        }

        cv::Rect best = faces[0];
        for (const auto& r : faces) if (r.area() > best.area()) best = r;

        cv::Rect clipped = best & cv::Rect(0, 0, bgr.cols, bgr.rows);
        if (clipped.area() <= 0) {
            err = "BBox volto non valida (cascade)";
            return false;
        }

        face_out = bgr(clipped).clone();
        return true;
    }

    #if CV_VERSION_MAJOR >= 4
    if (cfg.detector == "yunet") {
        if (cfg.detect_yunet.empty()) {
            err = "detector=yunet ma detect_yunet è vuoto";
            return false;
        }

        cv::Size inputSize(bgr.cols, bgr.rows);
        auto yn = cv::FaceDetectorYN::create(cfg.detect_yunet, "", inputSize, 0.9f, 0.3f, 5000);

        cv::Mat faces;
        yn->detect(bgr, faces);

        if (faces.empty() || faces.rows <= 0) {
            err = "Nessun volto (yunet)";
            return false;
        }

        int besti = 0;
        float bestArea = 0.f;
        for (int i = 0; i < faces.rows; ++i) {
            float w = faces.at<float>(i, 2);
            float h = faces.at<float>(i, 3);
            float area = w * h;
            if (area > bestArea) { bestArea = area; besti = i; }
        }

        int x = (int)faces.at<float>(besti, 0);
        int y = (int)faces.at<float>(besti, 1);
        int w = (int)faces.at<float>(besti, 2);
        int h = (int)faces.at<float>(besti, 3);

        cv::Rect r(x, y, w, h);
        r &= cv::Rect(0, 0, bgr.cols, bgr.rows);
        if (r.area() <= 0) {
            err = "BBox volto non valida (yunet)";
            return false;
        }

        face_out = bgr(r).clone();
        return true;
    }
    #endif

    err = "Detector non supportato: " + cfg.detector;
    return false;
}

extern "C" {

    bool fa_check_root(const std::string& tool_name) {
        if (geteuid() != 0) {
            std::cerr << "Errore: " << tool_name << " richiede privilegi di root.\n";
            return false;
        }
        return true;
    }

    bool fa_load_config(FacialAuthConfig& cfg, std::string& log, const std::string& path) {
        const std::string real_path = path.empty() ? default_config_path() : path;

        std::ifstream file(real_path);
        if (!file.is_open()) {
            log = "Config non trovata in " + real_path + " (uso default)";
            return true;
        }

        std::string line;
        while (std::getline(file, line)) {
            line = trim(line);
            if (line.empty() || line[0] == '#') continue;

            auto sep = line.find('=');
            if (sep == std::string::npos) continue;

            std::string key = trim(line.substr(0, sep));
            std::string val = trim(line.substr(sep + 1));

            // Paths
            if (key == "basedir") cfg.basedir = val;

            // Capture
            else if (key == "device") cfg.device = val;
            else if (key == "width") cfg.width = std::stoi(val);
            else if (key == "height") cfg.height = std::stoi(val);
            else if (key == "frames") cfg.frames = std::stoi(val);
            else if (key == "sleep_ms") cfg.sleep_ms = std::stoi(val);
            else if (key == "image_format") cfg.image_format = val;

            // Detector
            else if (key == "detector") cfg.detector = val;
            else if (key == "cascade_path") cfg.cascade_path = val;
            else if (key == "detect_yunet") cfg.detect_yunet = val;

            // Recognizer
            else if (key == "method") cfg.m
