/*
 * Project: pam_facial_auth
 * License: GPL-3.0
 *
 * Security requirement:
 *   Model always stored in /etc/security/pam_facial_auth/<user>.xml
 *   Directory permissions enforced (0700) and file permissions enforced (0600).
 */

#include "libfacialauth.h"

#include <cerrno>
#include <chrono>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>

#include <sys/stat.h>
#include <unistd.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/videoio.hpp>

#if CV_VERSION_MAJOR >= 4
#include <opencv2/objdetect/face.hpp> // FaceDetectorYN
#endif

namespace fs = std::filesystem;

// Factories implemented in plugin_*.cpp
std::unique_ptr<RecognizerPlugin> create_classic_plugin(const std::string& method, const FacialAuthConfig& cfg);
std::unique_ptr<RecognizerPlugin> create_sface_plugin(const FacialAuthConfig& cfg);

static std::string trim(std::string s) {
    auto issp = [](unsigned char c) { return std::isspace(c); };
    while (!s.empty() && issp((unsigned char)s.front())) s.erase(s.begin());
    while (!s.empty() && issp((unsigned char)s.back())) s.pop_back();
    return s;
}

static bool parse_bool(const std::string& v) {
    std::string s = v;
    for (auto& c : s) c = (char)std::tolower((unsigned char)c);
    return (s == "1" || s == "true" || s == "yes" || s == "on");
}

static bool parse_int(const std::string& v, int& out) {
    char* end = nullptr;
    errno = 0;
    long value = std::strtol(v.c_str(), &end, 10);
    if (errno != 0 || end == v.c_str() || *end != '\0') return false;
    out = static_cast<int>(value);
    return true;
}

static bool parse_double(const std::string& v, double& out) {
    char* end = nullptr;
    errno = 0;
    double value = std::strtod(v.c_str(), &end);
    if (errno != 0 || end == v.c_str() || *end != '\0') return false;
    out = value;
    return true;
}

static std::string default_config_path() {
    return "/etc/security/pam_facial_auth.conf";
}

// HARD security path
static std::string model_base_dir() {
    return "/etc/security/pam_facial_auth";
}

static std::string user_capture_dir(const FacialAuthConfig& cfg, const std::string& user) {
    return cfg.basedir + "/images/" + user;
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
        for (const auto& r : faces) {
            if (r.area() > best.area()) best = r;
        }

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
            if (area > bestArea) {
                bestArea = area;
                besti = i;
            }
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
        else if (key == "cascade_path") cfg.cascade_path = val;
        else if (key == "detect_yunet") cfg.detect_yunet = val;
        else if (key == "recognize_sface") cfg.recognize_sface = val;

        // Capture
        else if (key == "device") cfg.device = val;
        else if (key == "width") parse_int(val, cfg.width);
        else if (key == "height") parse_int(val, cfg.height);
        else if (key == "frames") parse_int(val, cfg.frames);
        else if (key == "sleep_ms") parse_int(val, cfg.sleep_ms);
        else if (key == "image_format") cfg.image_format = val;

        // Detector / Recognizer
        else if (key == "detector") cfg.detector = val;
        else if (key == "method") cfg.method = val;

        // Thresholds
        else if (key == "threshold") parse_double(val, cfg.threshold);
        else if (key == "lbph_threshold") parse_double(val, cfg.lbph_threshold);
        else if (key == "eigen_threshold") parse_double(val, cfg.eigen_threshold);
        else if (key == "fisher_threshold") parse_double(val, cfg.fisher_threshold);
        else if (key == "sface_threshold") parse_double(val, cfg.sface_threshold);

        // Flags
        else if (key == "debug") cfg.debug = parse_bool(val);
        else if (key == "verbose") cfg.verbose = parse_bool(val);
        else if (key == "nogui") cfg.nogui = parse_bool(val);
        else if (key == "ignore_failure") cfg.ignore_failure = parse_bool(val);
    }

    log = "Config caricata da " + real_path;
    return true;
}

std::string fa_user_model_path(const FacialAuthConfig& /*cfg*/, const std::string& user) {
    return model_base_dir() + "/" + user + ".xml";
}

bool fa_clean_captures(const std::string& user, const FacialAuthConfig& cfg, std::string& log) {
    const std::string dir = user_capture_dir(cfg, user);
    std::error_code ec;
    fs::remove_all(dir, ec);
    if (ec) {
        log = "Errore rimozione " + dir + ": " + ec.message();
        return false;
    }
    log = "Pulizia completata: " + dir;
    return true;
}

bool fa_capture_user(const std::string& user,
                     const FacialAuthConfig& cfg,
                     const std::string& device_path,
                     std::string& log) {
    const std::string dir = user_capture_dir(cfg, user);
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) {
        log = "Impossibile creare directory " + dir + ": " + ec.message();
        return false;
    }

    cv::VideoCapture cap(device_path);
    if (!cap.isOpened()) {
        log = "Impossibile aprire device " + device_path;
        return false;
    }
    if (cfg.width > 0) cap.set(cv::CAP_PROP_FRAME_WIDTH, cfg.width);
    if (cfg.height > 0) cap.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.height);

    int saved = 0;
    for (int i = 0; i < cfg.frames; ++i) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            continue;
        }

        cv::Mat face;
        std::string err;
        if (!detect_one_face(cfg, frame, face, err)) {
            if (cfg.debug) std::cerr << "[DEBUG] " << err << "\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(cfg.sleep_ms));
            continue;
        }

        std::string filename = dir + "/" + user + "_" + std::to_string(saved) + "." + cfg.image_format;
        if (!cv::imwrite(filename, face)) {
            log = "Impossibile salvare " + filename;
            return false;
        }

        ++saved;
        if (cfg.sleep_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(cfg.sleep_ms));
        }
    }

    if (saved == 0) {
        log = "Nessuna immagine catturata";
        return false;
    }

    log = "Catturate " + std::to_string(saved) + " immagini in " + dir;
    return true;
}

bool fa_train_user(const std::string& user, const FacialAuthConfig& cfg, std::string& log) {
    const std::string dir = user_capture_dir(cfg, user);
    std::vector<cv::Mat> faces;

    if (!fs::exists(dir)) {
        log = "Directory immagini non trovata: " + dir;
        return false;
    }

    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        cv::Mat img = cv::imread(entry.path().string());
        if (img.empty()) continue;

        cv::Mat face;
        std::string err;
        if (!detect_one_face(cfg, img, face, err)) {
            if (cfg.debug) std::cerr << "[DEBUG] " << err << "\n";
            continue;
        }
        faces.push_back(face);
    }

    if (faces.empty()) {
        log = "Nessuna immagine valida per il training in " + dir;
        return false;
    }

    std::string method = cfg.method;
    if (method == "auto" || method.empty()) {
        method = cfg.recognize_sface.empty() ? "lbph" : "sface";
    }

    auto plugin = create_plugin_for_method(cfg, method);
    if (!plugin) {
        log = "Impossibile creare plugin per metodo " + method;
        return false;
    }

    std::vector<int> labels(faces.size(), 1);
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

    log = "Modello creato: " + model_path + " (" + plugin->get_name() + ")";
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
    if (method == "auto" || method.empty()) {
        std::string alg = model_algorithm_from_xml(model_path);
        method = alg.empty() ? "lbph" : alg;
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
        log = "Impossibile aprire device " + cfg.device;
        return false;
    }

    double best_confidence = (plugin->get_name() == "sface") ? -1e9 : 1e9;
    int best_label = -1;
    bool any_predict = false;

    for (int i = 0; i < cfg.frames; ++i) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) continue;

        cv::Mat face;
        if (!detect_one_face(cfg, frame, face, err)) {
            if (cfg.debug) std::cerr << "[DEBUG] " << err << "\n";
            if (cfg.sleep_ms > 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(cfg.sleep_ms));
            }
            continue;
        }

        double conf = 0.0;
        int lbl = -1;
        if (!plugin->predict(face, lbl, conf, err)) {
            if (cfg.debug) std::cerr << "[DEBUG] " << err << "\n";
            if (cfg.sleep_ms > 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(cfg.sleep_ms));
            }
            continue;
        }

        any_predict = true;
        if (plugin->get_name() == "sface") {
            if (conf > best_confidence) {
                best_confidence = conf;
                best_label = lbl;
            }
        } else {
            if (conf < best_confidence) {
                best_confidence = conf;
                best_label = lbl;
            }
        }

        if (plugin->is_match(conf, cfg)) {
            confidence = conf;
            label = lbl;
            log = "Match OK per utente " + user + " (" + plugin->get_name() + ")";
            return true;
        }

        if (cfg.sleep_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(cfg.sleep_ms));
        }
    }

    if (any_predict) {
        confidence = best_confidence;
        label = best_label;
    }

    log = "Match fallito per utente " + user + " (" + plugin->get_name() + ")";
    return false;
}

} // extern "C"
