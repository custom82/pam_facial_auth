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

// --- Plugin LBPH / Eigen / Fisher ---
class ClassicPlugin : public RecognizerPlugin {
    cv::Ptr<cv::face::FaceRecognizer> model;
    std::string type;
    double threshold;
public:
    ClassicPlugin(const std::string& method, const FacialAuthConfig& cfg) : type(method) {
        if (method == "lbph") {
            model = cv::face::LBPHFaceRecognizer::create();
            threshold = cfg.lbph_threshold;
        } else if (method == "eigen") {
            model = cv::face::EigenFaceRecognizer::create(cfg.eigen_components);
            threshold = cfg.eigen_threshold;
        } else {
            model = cv::face::FisherFaceRecognizer::create(cfg.fisher_components);
            threshold = cfg.fisher_threshold;
        }
    }

    bool load(const std::string& path) override {
        if (!fs::exists(path)) return false;
        try {
            model->read(path);
            return true;
        } catch (...) { return false; }
    }

    bool train(const std::vector<cv::Mat>& faces, const std::vector<int>& labels, const std::string& save_path) override {
        if (faces.empty()) return false;
        std::vector<cv::Mat> grays;
        for (const auto& f : faces) {
            cv::Mat g;
            if (f.channels() == 3) cv::cvtColor(f, g, cv::COLOR_BGR2GRAY); else g = f.clone();
            grays.push_back(g);
        }
        model->train(grays, labels);
        model->save(save_path);
        return true;
    }

    bool predict(const cv::Mat& face, int& label, double& confidence) override {
        cv::Mat gray;
        if (face.channels() == 3) cv::cvtColor(face, gray, cv::COLOR_BGR2GRAY); else gray = face;
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

    bool load(const std::string& path) override {
        target_embeddings.clear();
        std::ifstream in(path, std::ios::binary);
        if (!in) return false;
        int count = 0;
        in.read((char*)&count, sizeof(count));
        for (int i = 0; i < count; ++i) {
            int r, c, t;
            in.read((char*)&r, sizeof(r)); in.read((char*)&c, sizeof(c)); in.read((char*)&t, sizeof(t));
            cv::Mat m(r, c, t);
            in.read((char*)m.data, m.total() * m.elemSize());
            target_embeddings.push_back(m);
        }
        return !target_embeddings.empty();
    }

    bool train(const std::vector<cv::Mat>& faces, const std::vector<int>& labels, const std::string& save_path) override {
        std::vector<cv::Mat> embeds;
        for (const auto& f : faces) {
            cv::Mat em;
            sface->feature(f, em);
            embeds.push_back(em.clone());
        }
        std::ofstream out(save_path, std::ios::binary);
        int count = embeds.size();
        out.write((char*)&count, sizeof(count));
        for (auto& m : embeds) {
            int r = m.rows, c = m.cols, t = m.type();
            out.write((char*)&r, sizeof(r)); out.write((char*)&c, sizeof(c)); out.write((char*)&t, sizeof(t));
            out.write((char*)m.data, m.total() * m.elemSize());
        }
        return true;
    }

    bool predict(const cv::Mat& face, int& label, double& confidence) override {
        cv::Mat query;
        sface->feature(face, query);
        double max_sim = -1.0;
        for (const auto& target : target_embeddings) {
            double sim = sface->match(query, target, cv::FaceRecognizerSF::DisType::FR_COSINE);
            if (sim > max_sim) max_sim = sim;
        }
        confidence = max_sim;
        label = (confidence >= threshold) ? 0 : -1;
        return (label == 0);
    }
    std::string get_name() const override { return "sface"; }
};

// --- API Implementation ---

std::unique_ptr<RecognizerPlugin> fa_create_plugin(const FacialAuthConfig& cfg) {
    std::string method = cfg.training_method;
    if (method == "auto") method = fs::exists(cfg.recognize_sface) ? "sface" : "lbph";
    if (method == "sface") return std::make_unique<SFacePlugin>(cfg);
    return std::make_unique<ClassicPlugin>(method, cfg);
}

bool fa_load_config(FacialAuthConfig &cfg, std::string &log, const std::string &path) {
    std::ifstream file(path);
    if (!file.is_open()) return false;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream is_line(line);
        std::string key, value;
        if (std::getline(is_line, key, '=') && std::getline(is_line, value)) {
            key.erase(key.find_last_not_of(" \t\n\r") + 1);
            value.erase(0, value.find_first_not_of(" \t\n\r"));
            if (key == "basedir") cfg.basedir = value;
            else if (key == "device") cfg.device = value;
            else if (key == "training_method") cfg.training_method = value;
            else if (key == "debug") cfg.debug = (value == "yes");
            else if (key == "ignore_failure") cfg.ignore_failure = (value == "yes");
        }
    }
    return true;
}

bool fa_train_user(const std::string &user, const FacialAuthConfig &cfg, std::string &log) {
    auto plugin = fa_create_plugin(cfg);
    std::vector<cv::Mat> faces;
    std::vector<int> labels;
    std::string path = cfg.basedir + "/" + user + "/captures";
    if (!fs::exists(path)) return false;
    for (const auto& entry : fs::directory_iterator(path)) {
        cv::Mat img = cv::imread(entry.path().string());
        if (!img.empty()) { faces.push_back(img); labels.push_back(0); }
    }
    return plugin->train(faces, labels, fa_user_model_path(cfg, user));
}

bool fa_test_user(const std::string &user, const FacialAuthConfig &cfg, const std::string &modelPath,
                  double &best_conf, int &best_label, std::string &log, double threshold_override) {
    auto plugin = fa_create_plugin(cfg);
    if (!plugin->load(modelPath)) return false;
    cv::VideoCapture cap;
    try { cap.open(std::stoi(cfg.device)); } catch(...) { cap.open(cfg.device); }
    if (!cap.isOpened()) return false;
    cv::Mat frame; cap >> frame;
    if (frame.empty()) return false;
    return plugin->predict(frame, best_label, best_conf);
                  }

                  std::string fa_user_model_path(const FacialAuthConfig &cfg, const std::string &user) {
                      return cfg.basedir + "/" + user + "/model" + (cfg.training_method == "sface" ? ".bin" : ".xml");
                  }

                  bool fa_file_exists(const std::string &path) { return fs::exists(path); }
                  bool fa_check_root(const std::string &t) { return getuid() == 0; }
