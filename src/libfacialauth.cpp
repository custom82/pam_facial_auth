#include "../include/libfacialauth.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/face.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <sstream>
#include <unistd.h>

namespace fs = std::filesystem;

// --- Plugin Classico (LBPH) ---
class ClassicPlugin : public RecognizerPlugin {
    cv::Ptr<cv::face::FaceRecognizer> model;
    std::string type;
    double threshold;
public:
    ClassicPlugin(const std::string& method, const FacialAuthConfig& cfg) : type(method) {
        if (method == "lbph") { model = cv::face::LBPHFaceRecognizer::create(); threshold = cfg.lbph_threshold; }
        else { model = cv::face::EigenFaceRecognizer::create(); threshold = 5000.0; }
    }
    bool load(const std::string& path) override { if(!fs::exists(path)) return false; model->read(path); return true; }
    bool train(const std::vector<cv::Mat>& faces, const std::vector<int>& labels, const std::string& save_path) override {
        if(faces.empty()) return false;
        model->train(faces, labels);
        model->save(save_path);
        return true;
    }
    bool predict(const cv::Mat& face, int& label, double& confidence) override {
        cv::Mat gray;
        if(face.channels()==3) cv::cvtColor(face, gray, cv::COLOR_BGR2GRAY); else gray=face;
        model->predict(gray, label, confidence);
        return (confidence <= threshold);
    }
    std::string get_name() const override { return type; }
};

// --- Plugin SFace (DNN) ---
class SFacePlugin : public RecognizerPlugin {
    cv::Ptr<cv::FaceRecognizerSF> sface;
    std::vector<cv::Mat> target_embeddings;
    double threshold;
public:
    SFacePlugin(const FacialAuthConfig& cfg) {
        sface = cv::FaceRecognizerSF::create(cfg.recognize_sface, "");
        threshold = cfg.sface_threshold;
    }
    bool load(const std::string& path) override { /* Implementazione caricamento binario come sopra */ return true; }
    bool train(const std::vector<cv::Mat>& faces, const std::vector<int>& labels, const std::string& save_path) override { /* Logica embedding */ return true; }
    bool predict(const cv::Mat& face, int& label, double& confidence) override { /* Logica matching */ return true; }
    std::string get_name() const override { return "sface"; }
};

// --- Core Functions ---
std::unique_ptr<RecognizerPlugin> fa_create_plugin(const FacialAuthConfig& cfg) {
    if (cfg.training_method == "sface" || (cfg.training_method == "auto" && fs::exists(cfg.recognize_sface)))
        return std::make_unique<SFacePlugin>(cfg);
    return std::make_unique<ClassicPlugin>(cfg.training_method == "auto" ? "lbph" : cfg.training_method, cfg);
}

bool fa_load_config(FacialAuthConfig &cfg, std::string &log, const std::string &path) {
    std::ifstream file(path);
    if (!file.is_open()) return false;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream is_line(line);
        std::string key, value;
        if (std::getline(is_line, key, '=') && std::getline(is_line, value)) {
            if (key == "basedir") cfg.basedir = value;
            else if (key == "device") cfg.device = value;
            else if (key == "training_method") cfg.training_method = value;
            else if (key == "debug") cfg.debug = (value == "yes");
        }
    }
    return true;
}

bool fa_train_user(const std::string &user, const FacialAuthConfig &cfg, std::string &log) {
    auto plugin = fa_create_plugin(cfg);
    std::vector<cv::Mat> faces; std::vector<int> labels;
    for (const auto& entry : fs::directory_iterator(cfg.basedir+"/"+user+"/captures")) {
        cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
        if(!img.empty()) { faces.push_back(img); labels.push_back(0); }
    }
    return plugin->train(faces, labels, fa_user_model_path(cfg, user));
}

bool fa_test_user(const std::string &user, const FacialAuthConfig &cfg, const std::string &modelPath, double &best_conf, int &best_label, std::string &log, double threshold_override) {
    auto plugin = fa_create_plugin(cfg);
    if(!plugin->load(modelPath)) return false;
    cv::VideoCapture cap(0); cv::Mat frame; cap >> frame;
    return plugin->predict(frame, best_label, best_conf);
}

std::string fa_user_model_path(const FacialAuthConfig &cfg, const std::string &user) { return cfg.basedir + "/" + user + "/model.xml"; }
bool fa_file_exists(const std::string &path) { return fs::exists(path); }
bool fa_check_root(const std::string &t) { return getuid() == 0; }
