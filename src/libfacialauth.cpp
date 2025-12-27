#include "../include/libfacialauth.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/face.hpp>
#include <opencv2/dnn.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unistd.h>

namespace fs = std::filesystem;

// --- Implementazione Plugin LBPH/Classic ---
class ClassicPlugin : public RecognizerPlugin {
    cv::Ptr<cv::face::FaceRecognizer> model;
public:
    ClassicPlugin(const std::string& method) {
        if (method == "eigen") model = cv::face::EigenFaceRecognizer::create();
        else if (method == "fisher") model = cv::face::FisherFaceRecognizer::create();
        else model = cv::face::LBPHFaceRecognizer::create();
    }
    bool load(const std::string& path) override { if(!fs::exists(path)) return false; model->read(path); return true; }
    bool train(const std::vector<cv::Mat>& f, const std::vector<int>& l, const std::string& p) override {
        model->train(f, l); model->save(p); return true;
    }
    bool predict(const cv::Mat& face, int& label, double& conf) override {
        cv::Mat gray;
        if(face.channels() == 3) cv::cvtColor(face, gray, cv::COLOR_BGR2GRAY); else gray = face;
        model->predict(gray, label, conf); return true;
    }
};

// Factory per i plugin
std::unique_ptr<RecognizerPlugin> fa_create_plugin(const FacialAuthConfig& cfg) {
    return std::make_unique<ClassicPlugin>(cfg.training_method);
}

// --- API Implementation ---

bool fa_load_config(FacialAuthConfig &cfg, std::string &log, const std::string &path) {
    std::ifstream file(path);
    if (!file.is_open()) { log = "Configurazione non trovata: " + path; return false; }
    // Qui andrebbe il parsing reale del file .conf
    return true;
}

bool fa_check_root(const std::string &tool_name) {
    if (getuid() != 0) {
        std::cerr << "Errore: " << tool_name << " deve essere eseguito come root.\n";
        return false;
    }
    return true;
}

std::string fa_user_model_path(const FacialAuthConfig &cfg, const std::string &user) {
    return cfg.basedir + "/" + user + "/model.xml";
}

bool fa_capture_user(const std::string &user, const FacialAuthConfig &cfg, const std::string &detector_type, std::string &log) {
    std::string user_dir = cfg.basedir + "/" + user + "/captures";
    if (cfg.force) fs::remove_all(user_dir);
    fs::create_directories(user_dir);

    cv::Ptr<cv::FaceDetectorYN> detector;
    if (detector_type == "yunet") {
        detector = cv::FaceDetectorYN::create(cfg.detect_model_path, "", cv::Size(320, 320));
    }

    cv::VideoCapture cap;
    try { cap.open(std::stoi(cfg.device)); } catch(...) { cap.open(cfg.device); }

    if (!cap.isOpened()) { log = "Impossibile aprire la camera"; return false; }

    int count = 0;
    while (count < cfg.frames) {
        cv::Mat frame; cap >> frame;
        if (frame.empty()) break;

        bool found = true;
        if (detector) {
            cv::Mat faces;
            detector->setInputSize(frame.size());
            detector->detect(frame, faces);
            found = (faces.rows > 0);
        }

        if (found) {
            std::string path = user_dir + "/img_" + std::to_string(count++) + "." + cfg.image_format;
            cv::imwrite(path, frame);
        }

        if (!cfg.nogui) {
            cv::imshow("Cattura - " + user, frame);
            if (cv::waitKey(1) == 'q') break;
        }
    }
    cv::destroyAllWindows();
    return true;
}

bool fa_train_user(const std::string &user, const FacialAuthConfig &cfg, std::string &log) {
    std::string user_dir = cfg.basedir + "/" + user + "/captures";
    std::vector<cv::Mat> faces;
    std::vector<int> labels;

    if (!fs::exists(user_dir)) { log = "Nessun sample trovato per l'utente"; return false; }

    for (const auto& entry : fs::directory_iterator(user_dir)) {
        cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
        if (!img.empty()) {
            faces.push_back(img);
            labels.push_back(0); // 0 è l'ID dell'utente corrente
        }
    }

    if (faces.empty()) { log = "Nessuna immagine valida caricata"; return false; }

    auto plugin = fa_create_plugin(cfg);
    return plugin->train(faces, labels, fa_user_model_path(cfg, user));
}

bool fa_test_user_interactive(const std::string &user, const FacialAuthConfig &cfg, std::string &log) {
    auto plugin = fa_create_plugin(cfg);
    if (!plugin->load(fa_user_model_path(cfg, user))) { log = "Modello non trovato"; return false; }

    cv::VideoCapture cap;
    try { cap.open(std::stoi(cfg.device)); } catch(...) { cap.open(cfg.device); }

    while (true) {
        cv::Mat frame; cap >> frame;
        if (frame.empty()) break;

        int label = -1;
        double confidence = 0.0;
        plugin->predict(frame, label, confidence);

        std::string text = "User: " + (label == 0 ? user : "Sconosciuto") + " (" + std::to_string(confidence) + ")";
        cv::putText(frame, text, {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {0, 255, 0}, 2);

        cv::imshow("Test Interattivo", frame);
        if (cv::waitKey(1) == 'q') break;
    }
    cv::destroyAllWindows();
    return true;
}
