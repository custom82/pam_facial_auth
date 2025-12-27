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
#include <cstdlib>
#include <regex>

namespace fs = std::filesystem;

/**
 * Implementation of Classic OpenCV Face Recognizers
 */
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
        try { model->read(path); return true; } catch (...) { return false; }
    }
    bool train(const std::vector<cv::Mat>& faces, const std::vector<int>& labels, const std::string& save_path) override {
        if (faces.empty()) return false;
        try { model->train(faces, labels); model->save(save_path); return true; } catch (...) { return false; }
    }
    bool predict(const cv::Mat& face, int& label, double& confidence) override {
        cv::Mat gray;
        if (face.channels() == 3) cv::cvtColor(face, gray, cv::COLOR_BGR2GRAY);
        else gray = face;
        try { model->predict(gray, label, confidence); return true; } catch (...) { return false; }
    }
};

bool fa_file_exists(const std::string &path) { return fs::exists(path); }

bool fa_load_config(FacialAuthConfig &cfg, std::string &log, const std::string &path) {
    if (!fs::exists(path)) { log = "Config not found, using defaults."; return false; }
    return true;
}

bool fa_check_root(const std::string &tool_name) {
    if (getuid() != 0) { std::cerr << "Error: " << tool_name << " must be run as root.\n"; return false; }
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

/**
 * Capture user faces with auto-incrementing filenames and face detection filtering
 */
bool fa_capture_user(const std::string &user, const FacialAuthConfig &cfg, const std::string &det_type, std::string &log) {
    std::string user_dir_str = cfg.basedir + "/" + user + "/captures";
    fs::path user_dir(user_dir_str);

    // If --force is set, we wipe everything and start from 0
    if (cfg.force) {
        fs::remove_all(user_dir);
    }
    fs::create_directories(user_dir);

    // AUTO-INCREMENT LOGIC: Find the highest 'img_X.ext' to resume counting
    int start_index = 0;
    if (!cfg.force && fs::exists(user_dir)) {
        // Regex to match "img_" followed by digits and the selected extension
        std::regex file_regex("img_([0-9]+)\\." + cfg.image_format);
        for (const auto& entry : fs::directory_iterator(user_dir)) {
            std::string filename = entry.path().filename().string();
            std::smatch match;
            if (std::regex_match(filename, match, file_regex)) {
                try {
                    int index = std::stoi(match[1].str());
                    if (index >= start_index) start_index = index + 1;
                } catch (...) { continue; }
            }
        }
    }

    // Display check for Wayland/X11 safety
    bool display_available = (std::getenv("DISPLAY") != NULL || std::getenv("WAYLAND_DISPLAY") != NULL);
    bool should_show_gui = (!cfg.nogui && display_available);

    cv::Ptr<cv::FaceDetectorYN> yunet;
    cv::CascadeClassifier haar;

    if (det_type == "yunet") {
        if (!fs::exists(cfg.detect_model_path)) {
            log = "YuNet model NOT FOUND at: " + cfg.detect_model_path;
            return false;
        }
        try {
            yunet = cv::FaceDetectorYN::create(cfg.detect_model_path, "", cv::Size(cfg.width, cfg.height));
        } catch (const cv::Exception& e) {
            log = "OpenCV DNN Error: " + std::string(e.what());
            return false;
        }
    } else if (det_type == "haar") {
        if (!haar.load(cfg.haar_path)) { log = "Haar XML not found at " + cfg.haar_path; return false; }
    }

    cv::VideoCapture cap;
    try { cap.open(std::stoi(cfg.device)); } catch(...) { cap.open(cfg.device); }
    if (!cap.isOpened()) { log = "Camera device error."; return false; }

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
            // Filename based on existing files + session progress
            std::string filename = "img_" + std::to_string(start_index + count) + "." + cfg.image_format;
            fs::path full_path = user_dir / filename;
            cv::imwrite(full_path.string(), frame);

            // Output always active as requested
            std::cout << "Saved: " << fs::absolute(full_path).string() << std::endl;
            count++;
        } else if (cfg.debug) {
            std::cout << "[DEBUG] No face detected in frame, skipping." << std::endl;
        }

        if (should_show_gui) {
            try {
                cv::imshow("Facial Auth Capture", frame);
                if (cv::waitKey(1) == 'q') break;
            } catch (...) { should_show_gui = false; }
        }
    }
    if (should_show_gui) cv::destroyAllWindows();
    return (count >= cfg.frames);
}

bool fa_train_user(const std::string &user, const FacialAuthConfig &cfg, std::string &log) {
    std::string user_dir = cfg.basedir + "/" + user + "/captures";
    std::vector<cv::Mat> faces; std::vector<int> labels;
    if (!fs::exists(user_dir)) { log = "Capture directory missing."; return false; }
    for (const auto& entry : fs::directory_iterator(user_dir)) {
        cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
        if (!img.empty()) { faces.push_back(img); labels.push_back(0); }
    }
    ClassicPlugin plugin(cfg.training_method);
    return plugin.train(faces, labels, fa_user_model_path(cfg, user));
}

bool fa_test_user(const std::string &user, const FacialAuthConfig &cfg, const std::string &model_path, double &conf, int &label, std::string &log) {
    ClassicPlugin plugin(cfg.training_method);
    if (!plugin.load(model_path)) { log = "Model load failed."; return false; }
    cv::VideoCapture cap;
    try { cap.open(std::stoi(cfg.device)); } catch(...) { cap.open(cfg.device); }
    cv::Mat frame; cap >> frame;
    if (frame.empty()) return false;
    return plugin.predict(frame, label, conf);
}
