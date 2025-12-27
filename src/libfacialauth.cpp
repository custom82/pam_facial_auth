#include "../include/libfacialauth.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/face.hpp>
#include <opencv2/dnn.hpp>
#include <filesystem>
#include <iostream>
#include <unistd.h>

namespace fs = std::filesystem;

class ClassicPlugin : public RecognizerPlugin {
    cv::Ptr<cv::face::FaceRecognizer> model;
public:
    ClassicPlugin(const std::string& method) {
        if (method == "eigen") model = cv::face::EigenFaceRecognizer::create();
        else if (method == "fisher") model = cv::face::FisherFaceRecognizer::create();
        else model = cv::face::LBPHFaceRecognizer::create();
    }
    bool load(const std::string& path) override {
        if (!fs::exists(path)) return false;
        model->read(path);
        return true;
    }
    bool train(const std::vector<cv::Mat>& faces, const std::vector<int>& labels, const std::string& save_path) override {
        if (faces.empty()) return false;
        model->train(faces, labels);
        model->save(save_path);
        return true;
    }
    bool predict(const cv::Mat& face, int& label, double& confidence) override {
        cv::Mat gray;
        if (face.channels() == 3) cv::cvtColor(face, gray, cv::COLOR_BGR2GRAY);
        else gray = face;
        model->predict(gray, label, confidence);
        return true;
    }
};

bool fa_file_exists(const std::string &path) { return fs::exists(path); }

bool fa_load_config(FacialAuthConfig &cfg, std::string &log, const std::string &path) {
    if (!fs::exists(path)) { log = "Config non trovata, uso default."; return false; }
    return true;
}

bool fa_check_root(const std::string &tool_name) {
    if (getuid() != 0) { std::cerr << "Errore: " << tool_name << " deve essere root.\n"; return false; }
    return true;
}

bool fa_delete_user_data(const std::string &user, const FacialAuthConfig &cfg) {
    std::string path = cfg.basedir + "/" + user;
    if (fs::exists(path)) { fs::remove_all(path); return true; }
    return false;
}

std::string fa_user_model_path(const FacialAuthConfig &cfg, const std::string &user) {
    return cfg.basedir + "/" + user + "/model.xml";
}

bool fa_capture_user(const std::string &user, const FacialAuthConfig &cfg, const std::string &det_type, std::string &log) {
    std::string user_dir = cfg.basedir + "/" + user + "/captures";
    if (cfg.force) fs::remove_all(user_dir);
    fs::create_directories(user_dir);

    cv::Ptr<cv::FaceDetectorYN> yunet;
    cv::CascadeClassifier haar;
    if (det_type == "yunet") {
        yunet = cv::FaceDetectorYN::create(cfg.detect_model_path, "", cv::Size(cfg.width, cfg.height));
    } else if (det_type == "haar") {
        if (!haar.load(cfg.haar_path)) { log = "Haar XML non trovato in " + cfg.haar_path; return false; }
    }

    cv::VideoCapture cap;
    try { cap.open(std::stoi(cfg.device)); } catch(...) { cap.open(cfg.device); }
    if (!cap.isOpened()) { log = "Camera error"; return false; }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, cfg.width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.height);

    int count = 0;
    while (count < cfg.frames) {
        cv::Mat frame; cap >> frame;
        if (frame.empty()) break;

        bool face_found = (det_type == "none");

        if (det_type == "yunet") {
            cv::Mat faces;
            yunet->setInputSize(frame.size());
            yunet->detect(frame, faces);
            face_found = (faces.rows > 0);
        } else if (det_type == "haar") {
            cv::Mat gray;
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            std::vector<cv::Rect> faces;
            haar.detectMultiScale(gray, faces, 1.1, 3);
            face_found = !faces.empty();
        }

        if (face_found) {
            std::string filename = user_dir + "/img_" + std::to_string(count++) + "." + cfg.image_format;
            cv::imwrite(filename, frame);
            if (cfg.debug) std::cout << "[DEBUG] " << count << "/" << cfg.frames << " salvato.\n";
        } else if (cfg.debug) {
            std::cout << "[DEBUG] Volto non rilevato, frame saltato.\n";
        }

        if (!cfg.nogui) {
            cv::imshow("Capture", frame);
            if (cv::waitKey(1) == 'q') break;
        }
    }
    cv::destroyAllWindows();
    return (count >= cfg.frames);
}

bool fa_train_user(const std::string &user, const FacialAuthConfig &cfg, std::string &log) {
    std::string user_dir = cfg.basedir + "/" + user + "/captures";
    std::vector<cv::Mat> faces; std::vector<int> labels;
    if (!fs::exists(user_dir)) return false;
    for (const auto& entry : fs::directory_iterator(user_dir)) {
        cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
        if (!img.empty()) { faces.push_back(img); labels.push_back(0); }
    }
    ClassicPlugin plugin(cfg.training_method);
    return plugin.train(faces, labels, fa_user_model_path(cfg, user));
}

bool fa_test_user(const std::string &user, const FacialAuthConfig &cfg, const std::string &model_path, double &conf, int &label, std::string &log) {
    ClassicPlugin plugin(cfg.training_method);
    if (!plugin.load(model_path)) return false;
    cv::VideoCapture cap;
    try { cap.open(std::stoi(cfg.device)); } catch(...) { cap.open(cfg.device); }
    cv::Mat frame; cap >> frame;
    return !frame.empty() && plugin.predict(frame, label, conf);
}
