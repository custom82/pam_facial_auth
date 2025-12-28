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
#include <chrono>
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
            else if (key == "method") cfg.method = val;
            else if (key == "recognize_sface") cfg.recognize_sface = val;

            // Thresholds
            else if (key == "threshold") cfg.threshold = std::stod(val);
            else if (key == "lbph_threshold") cfg.lbph_threshold = std::stod(val);
            else if (key == "eigen_threshold") cfg.eigen_threshold = std::stod(val);
            else if (key == "fisher_threshold") cfg.fisher_threshold = std::stod(val);
            else if (key == "sface_threshold") cfg.sface_threshold = std::stod(val);

            // Output / debug
            else if (key == "debug") cfg.debug = parse_bool(val);
            else if (key == "verbose") cfg.verbose = parse_bool(val);
            else if (key == "nogui") cfg.nogui = parse_bool(val);

            // PAM behavior
            else if (key == "ignore_failure") cfg.ignore_failure = parse_bool(val);
        }

        log = "Config caricata da " + real_path;
        return true;
    }

    std::string fa_user_model_path(const FacialAuthConfig& /*cfg*/, const std::string& user) {
        return model_base_dir() + "/" + user + ".xml";
    }

    bool fa_clean_captures(const std::string& user, const FacialAuthConfig& cfg, std::string& log) {
        const fs::path dir = fs::path(cfg.basedir) / "captures" / user;
        std::error_code ec;
        const std::uintmax_t removed = fs::remove_all(dir, ec);
        if (ec) {
            log = "Errore rimozione catture: " + dir.string() + " (" + ec.message() + ")";
            return false;
        }
        log = "Catture rimosse (" + std::to_string(removed) + " elementi) per " + user;
        return true;
    }

    bool fa_capture_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& device_path, std::string& log) {
        const fs::path out_dir = fs::path(cfg.basedir) / "captures" / user;
        std::error_code ec;
        fs::create_directories(out_dir, ec);
        if (ec) {
            log = "Impossibile creare directory catture: " + out_dir.string() + " (" + ec.message() + ")";
            return false;
        }

        cv::VideoCapture cap(device_path);
        if (!cap.isOpened()) {
            log = "Impossibile aprire dispositivo: " + device_path;
            return false;
        }

        cap.set(cv::CAP_PROP_FRAME_WIDTH, cfg.width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.height);

        int saved = 0;
        for (int i = 0; i < cfg.frames; ++i) {
            cv::Mat frame;
            cap >> frame;
            if (frame.empty()) continue;

            cv::Mat face;
            std::string err;
            if (!detect_one_face(cfg, frame, face, err)) {
                if (cfg.debug) std::cerr << "[DEBUG] frame " << i << ": " << err << "\n";
                std::this_thread::sleep_for(std::chrono::milliseconds(cfg.sleep_ms));
                continue;
            }

            const std::string filename = "f_" + std::to_string(i) + "." + cfg.image_format;
            const fs::path out_path = out_dir / filename;
            if (!cv::imwrite(out_path.string(), face)) {
                if (cfg.debug) std::cerr << "[DEBUG] Salvataggio fallito: " << out_path << "\n";
                std::this_thread::sleep_for(std::chrono::milliseconds(cfg.sleep_ms));
                continue;
            }

            ++saved;
            if (!cfg.nogui) {
                cv::imshow("pam_facial_auth capture", face);
                cv::waitKey(1);
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(cfg.sleep_ms));
        }

        log = "Catture completate: " + std::to_string(saved) + " immagini per " + user;
        return saved > 0;
    }

    bool fa_train_user(const std::string& user, const FacialAuthConfig& cfg, std::string& log) {
        const fs::path cap_dir = fs::path(cfg.basedir) / "captures" / user;
        if (!fs::exists(cap_dir)) {
            log = "Directory catture non trovata: " + cap_dir.string();
            return false;
        }

        std::vector<cv::Mat> faces;
        std::vector<int> labels;
        for (const auto& entry : fs::directory_iterator(cap_dir)) {
            if (!entry.is_regular_file()) continue;
            cv::Mat img = cv::imread(entry.path().string());
            if (img.empty()) continue;
            faces.push_back(img);
            labels.push_back(1);
        }

        if (faces.empty()) {
            log = "Nessuna immagine valida in " + cap_dir.string();
            return false;
        }

        std::string method = cfg.method;
        if (method == "auto") {
            method = cfg.recognize_sface.empty() ? "lbph" : "sface";
        }

        if (method == "sface" && cfg.recognize_sface.empty()) {
            log = "recognize_sface mancante per metodo sface";
            return false;
        }

        auto plugin = create_plugin_for_method(cfg, method);
        if (!plugin) {
            log = "Impossibile creare plugin per metodo " + method;
            return false;
        }

        std::string err;
        const std::string model_path = fa_user_model_path(cfg, user);
        if (!ensure_secure_model_dir(err)) {
            log = err;
            return false;
        }

        if (!plugin->train(faces, labels, model_path, err)) {
            log = "Training fallito: " + err;
            return false;
        }

        if (!chmod0600(model_path, err)) {
            log = err;
            return false;
        }

        log = "Training completato con metodo " + method + " (" + std::to_string(faces.size()) + " immagini)";
        return true;
    }

    bool fa_test_user(const std::string& user,
                      const FacialAuthConfig& cfg,
                      const std::string& model_path,
                      double& confidence,
                      int& label,
                      std::string& log) {
        if (!fs::exists(model_path)) {
            log = "Modello non trovato: " + model_path;
            return false;
        }

        std::string method = cfg.method;
        const std::string model_alg = model_algorithm_from_xml(model_path);
        if (!model_alg.empty()) method = model_alg;
        if (method == "auto") method = cfg.recognize_sface.empty() ? "lbph" : "sface";
        if (method == "sface" && cfg.recognize_sface.empty()) {
            log = "recognize_sface mancante per metodo sface";
            return false;
        }

        auto plugin = create_plugin_for_method(cfg, method);
        if (!plugin) {
            log = "Impossibile creare plugin per metodo " + method;
            return false;
        }

        std::string err;
        if (!plugin->load(model_path, err)) {
            log = "Caricamento modello fallito: " + err;
            return false;
        }

        cv::VideoCapture cap(cfg.device);
        if (!cap.isOpened()) {
            log = "Impossibile aprire dispositivo: " + cfg.device;
            return false;
        }

        cap.set(cv::CAP_PROP_FRAME_WIDTH, cfg.width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.height);

        bool has_prediction = false;
        double best_confidence = 0.0;
        int best_label = -1;

        for (int i = 0; i < cfg.frames; ++i) {
            cv::Mat frame;
            cap >> frame;
            if (frame.empty()) continue;

            cv::Mat face;
            if (!detect_one_face(cfg, frame, face, err)) {
                if (cfg.debug) std::cerr << "[DEBUG] frame " << i << ": " << err << "\n";
                std::this_thread::sleep_for(std::chrono::milliseconds(cfg.sleep_ms));
                continue;
            }

            double current_confidence = 0.0;
            int current_label = -1;
            if (!plugin->predict(face, current_label, current_confidence, err)) {
                if (cfg.debug) std::cerr << "[DEBUG] predict: " << err << "\n";
                std::this_thread::sleep_for(std::chrono::milliseconds(cfg.sleep_ms));
                continue;
            }

            has_prediction = true;
            best_confidence = current_confidence;
            best_label = current_label;

            if (plugin->is_match(current_confidence, cfg)) {
                confidence = current_confidence;
                label = current_label;
                log = "Match trovato con metodo " + method;
                return true;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(cfg.sleep_ms));
        }

        if (!has_prediction) {
            log = "Nessuna predizione valida";
            return false;
        }

        confidence = best_confidence;
        label = best_label;
        log = "Nessun match (metodo " + method + ")";
        return false;
    }

#ifdef __cplusplus
}
#endif
