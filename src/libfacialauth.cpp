#include "../include/libfacialauth.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/face.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;

// --- PLUGIN CLASSICO (LBPH/EIGEN/FISHER) ---
class ClassicPlugin : public RecognizerPlugin {
    cv::Ptr<cv::face::FaceRecognizer> model;
    std::string type; double threshold;
public:
    ClassicPlugin(const std::string& m, const FacialAuthConfig& cfg) : type(m) {
        if (m == "lbph") { model = cv::face::LBPHFaceRecognizer::create(); threshold = cfg.lbph_threshold; }
        else if (m == "eigen") { model = cv::face::EigenFaceRecognizer::create(cfg.eigen_components); threshold = cfg.eigen_threshold; }
        else { model = cv::face::FisherFaceRecognizer::create(cfg.fisher_components); threshold = cfg.fisher_threshold; }
    }
    bool load(const std::string& p) override { if(!fs::exists(p)) return false; model->read(p); return true; }
    bool train(const std::vector<cv::Mat>& f, const std::vector<int>& l, const std::string& s) override {
        std::vector<cv::Mat> grays;
        for(auto& img : f) { cv::Mat g; cv::cvtColor(img, g, cv::COLOR_BGR2GRAY); grays.push_back(g); }
        model->train(grays, l); model->save(s); return true;
    }
    bool predict(const cv::Mat& f, int& l, double& c) override {
        cv::Mat g; if(f.channels()==3) cv::cvtColor(f, g, cv::COLOR_BGR2GRAY); else g=f;
        model->predict(g, l, c); return (c <= threshold);
    }
    std::string get_name() const override { return type; }
};

// --- PLUGIN SFACE ---
class SFacePlugin : public RecognizerPlugin {
    cv::Ptr<cv::FaceRecognizerSF> sface;
    std::vector<cv::Mat> targets; double threshold;
public:
    SFacePlugin(const FacialAuthConfig& cfg) {
        sface = cv::FaceRecognizerSF::create(cfg.recognize_sface, "");
        threshold = cfg.sface_threshold;
    }
    bool load(const std::string& p) override {
        std::ifstream in(p, std::ios::binary); if(!in) return false;
        int count; in.read((char*)&count, sizeof(count));
        for(int i=0; i<count; ++i) {
            int r, c, t; in.read((char*)&r, sizeof(r)); in.read((char*)&c, sizeof(c)); in.read((char*)&t, sizeof(t));
            cv::Mat m(r, c, t); in.read((char*)m.data, m.total()*m.elemSize()); targets.push_back(m);
        }
        return !targets.empty();
    }
    bool train(const std::vector<cv::Mat>& f, const std::vector<int>& l, const std::string& s) override {
        std::ofstream out(s, std::ios::binary); int count = f.size(); out.write((char*)&count, sizeof(count));
        for(auto& img : f) {
            cv::Mat em; sface->feature(img, em);
            int r=em.rows, c=em.cols, t=em.type();
            out.write((char*)&r, sizeof(r)); out.write((char*)&c, sizeof(c)); out.write((char*)&t, sizeof(t));
            out.write((char*)em.data, em.total()*em.elemSize());
        }
        return true;
    }
    bool predict(const cv::Mat& f, int& l, double& c) override {
        cv::Mat q; sface->feature(f, q); double ms = -1.0;
        for(auto& t : targets) { double s = sface->match(q, t, cv::FaceRecognizerSF::DisType::FR_COSINE); if(s > ms) ms = s; }
        c = ms; l = (c >= threshold) ? 0 : -1; return (l == 0);
    }
    std::string get_name() const override { return "sface"; }
};

// Factory
std::unique_ptr<RecognizerPlugin> fa_create_plugin(const FacialAuthConfig& cfg) {
    std::string m = cfg.training_method;
    if (m == "auto") m = fs::exists(cfg.recognize_sface) ? "sface" : "lbph";
    if (m == "sface") return std::make_unique<SFacePlugin>(cfg);
    return std::make_unique<ClassicPlugin>(m, cfg);
}

// Implementazione fa_train_user (esempio con caricamento immagini)
bool fa_train_user(const std::string &user, const FacialAuthConfig &cfg, std::string &log) {
    auto plugin = fa_create_plugin(cfg);
    std::vector<cv::Mat> faces; std::vector<int> labels;
    std::string path = cfg.basedir + "/" + user + "/captures";
    if(!fs::exists(path)) return false;
    for(auto& p : fs::directory_iterator(path)) {
        cv::Mat img = cv::imread(p.path().string());
        if(!img.empty()) { faces.push_back(img); labels.push_back(0); }
    }
    return plugin->train(faces, labels, fa_user_model_path(cfg, user));
}

// Altre helper...
std::string fa_user_model_path(const FacialAuthConfig &cfg, const std::string &user) {
    return cfg.basedir + "/" + user + "/model.bin";
}
bool fa_file_exists(const std::string &path) { return fs::exists(path); }
