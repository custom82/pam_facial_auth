#include <string>
#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <chrono>
#include <thread>

namespace fs = std::filesystem;

extern "C" {
    #include <security/pam_modules.h>
    #include <security/pam_ext.h>
}

struct Config {
    std::string device = "/dev/video0";
    int width = 640;
    int height = 480;
    double threshold = 75.0;
    int timeout = 10;
    bool nogui = false;
    bool debug = false;
    std::string model_path = ""; // Optional: specify model path
};

// === Carica la configurazione base ===
Config load_config(const std::string &path = "/etc/pam_facial_auth/pam_facial.conf") {
    Config cfg;
    std::ifstream conf(path);
    if (!conf.is_open()) {
        std::cerr << "[WARN] pam_facial_auth: config file not found, using defaults\n";
        return cfg;
    }

    std::string key, value;
    while (conf >> key >> value) {
        if (key == "device") cfg.device = value;
        else if (key == "width") cfg.width = std::stoi(value);
        else if (key == "height") cfg.height = std::stoi(value);
        else if (key == "threshold") cfg.threshold = std::stod(value);
        else if (key == "timeout") cfg.timeout = std::stoi(value);
        else if (key == "nogui") cfg.nogui = (value == "true" || value == "1");
        else if (key == "debug") cfg.debug = (value == "true" || value == "1");
        else if (key == "model_path") cfg.model_path = value;  // New model_path option
    }

    return cfg;
}

// === Applica override dai parametri PAM ===
void apply_pam_args(Config &cfg, pam_handle_t *pamh, int argc, const char **argv) {
    for (int i = 0; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "nogui") cfg.nogui = true;
        else if (arg == "debug") cfg.debug = true;
        else if (arg.rfind("device=", 0) == 0) cfg.device = arg.substr(7);
        else if (arg.rfind("width=", 0) == 0) cfg.width = std::stoi(arg.substr(6));
        else if (arg.rfind("height=", 0) == 0) cfg.height = std::stoi(arg.substr(7));
        else if (arg.rfind("threshold=", 0) == 0) cfg.threshold = std::stod(arg.substr(10));
        else if (arg.rfind("timeout=", 0) == 0) cfg.timeout = std::stoi(arg.substr(8));
        else if (arg.rfind("model_path=", 0) == 0) cfg.model_path = arg.substr(11);  // Override model path from PAM args
    }

    if (cfg.debug) {
        pam_syslog(pamh, LOG_DEBUG,
                   "pam_facial_auth config: device=%s width=%d height=%d threshold=%.2f timeout=%d nogui=%d model_path=%s",
                   cfg.device.c_str(), cfg.width, cfg.height, cfg.threshold, cfg.timeout, cfg.nogui, cfg.model_path.c_str());
    }
}

extern "C" int pam_sm_authenticate(pam_handle_t *pamh, int flags, int argc, const char **argv) {
    const char *user = nullptr;
    if (pam_get_user(pamh, &user, nullptr) != PAM_SUCCESS || !user) {
        pam_syslog(pamh, LOG_ERR, "pam_facial_auth: unable to retrieve username");
        return PAM_AUTH_ERR;
    }

    // 1️⃣ Legge la configurazione
    Config cfg = load_config();
    apply_pam_args(cfg, pamh, argc, argv);

    // 2️⃣ Percorso del modello (usando il parametro manuale o quello di configurazione)
    std::string model_path;
    if (!cfg.model_path.empty()) {
        // Se model_path è specificato, usa la directory dell'utente
        model_path = cfg.model_path + "/" + std::string(user) + "/models/" + std::string(user) + ".xml";
    } else {
        // Se model_path è vuoto, cerca il modello nella directory dell'utente
        model_path = "/etc/pam_facial_auth/" + std::string(user) + "/models/" + std::string(user) + ".xml";
    }

    if (!fs::exists(model_path)) {
        pam_syslog(pamh, LOG_ERR, "pam_facial_auth: model not found for user %s at %s",
                   user, model_path.c_str());
        return PAM_AUTH_ERR;
    }

    // 3️⃣ Carica il modello facciale
    cv::Ptr<cv::face::LBPHFaceRecognizer> model = cv::face::LBPHFaceRecognizer::create();
    try {
        model->read(model_path);
    } catch (const cv::Exception &e) {
        pam_syslog(pamh, LOG_ERR, "pam_facial_auth: cannot load model: %s", e.what());
        return PAM_AUTH_ERR;
    }

    // 4️⃣ Apri webcam
    cv::VideoCapture cap(cfg.device);
    if (!cap.isOpened()) {
        pam_syslog(pamh, LOG_ERR, "pam_facial_auth: cannot open webcam %s", cfg.device.c_str());
        return PAM_AUTH_ERR;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, cfg.width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.height);

    if (cfg.debug)
        pam_syslog(pamh, LOG_DEBUG, "pam_facial_auth: webcam %s opened (%dx%d)",
                   cfg.device.c_str(), cfg.width, cfg.height);

        // 5️⃣ Cattura frame entro timeout
        cv::Mat frame;
    bool captured = false;
    auto start = std::chrono::steady_clock::now();
    while (!captured) {
        cap >> frame;
        if (!frame.empty()) captured = true;
        if (std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - start).count() >= cfg.timeout) {
            pam_syslog(pamh, LOG_ERR, "pam_facial_auth: timeout after %d seconds", cfg.timeout);
        return PAM_AUTH_ERR;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // 6️⃣ Converti in scala di grigi
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    int label = -1;
    double confidence = 0.0;
    try {
        model->predict(gray, label, confidence);
    } catch (const cv::Exception &e) {
        pam_syslog(pamh, LOG_ERR, "pam_facial_auth: prediction failed: %s", e.what());
        return PAM_AUTH_ERR;
    }

    if (cfg.debug)
        pam_syslog(pamh, LOG_DEBUG, "pam_facial_auth: confidence=%.2f threshold=%.2f",
                   confidence, cfg.threshold);

        if (confidence < cfg.threshold) {
            pam_syslog(pamh, LOG_INFO, "pam_facial_auth: authentication success for %s", user);
            return PAM_SUCCESS;
        } else {
            pam_syslog(pamh, LOG_WARNING,
                       "pam_facial_auth: authentication failed for %s (confidence %.2f ≥ %.2f)",
                       user, confidence, cfg.threshold);
            return PAM_AUTH_ERR;
        }
}

extern "C" int pam_sm_setcred(pam_handle_t *pamh, int flags, int argc, const char **argv) {
    (void)pamh; (void)flags; (void)argc; (void)argv;
    return PAM_SUCCESS;
}
