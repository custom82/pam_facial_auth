/*
 * Project: pam_facial_auth
 * License: GPL-3.0
 */

#include "libfacialauth.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <thread>
#include <regex>
#include <unistd.h>
#include <ctime>
#include <opencv2/face.hpp>

namespace fs = std::filesystem;

int get_last_index(const std::string& dir) {
    int max_idx = -1;
    if (!fs::exists(dir)) return max_idx;
    std::regex re("frame_(\\d+)\\.\\w+");
    for (const auto& entry : fs::directory_iterator(dir)) {
        std::smatch match;
        if (std::regex_match(entry.path().filename().string(), match, re)) {
            try { max_idx = std::max(max_idx, std::stoi(match[1].str())); } catch (...) {}
        }
    }
    return max_idx;
}

extern "C" {

    FA_EXPORT bool fa_check_root(const std::string& tool_name) {
        if (getuid() != 0) {
            std::cerr << "Errore: " << tool_name << " richiede privilegi di root." << std::endl;
            return false;
        }
        return true;
    }

    FA_EXPORT std::string fa_user_model_path(const FacialAuthConfig& cfg, const std::string& user) {
        return cfg.modeldir + "/" + user + ".xml";
    }

    FA_EXPORT bool fa_load_config(FacialAuthConfig& cfg, std::string& log, const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) { log = "Config mancante: " + path; return false; }
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            size_t sep = line.find('=');
            if (sep == std::string::npos) continue;
            std::string key = line.substr(0, sep), val = line.substr(sep + 1);
            key.erase(key.find_last_not_of(" \t") + 1);
            val.erase(0, val.find_first_not_of(" \t"));

            if (key == "basedir") cfg.basedir = val;
            else if (key == "device") cfg.device = val;
            else if (key == "detect_yunet") cfg.detect_yunet = val;
            else if (key == "recognize_sface") cfg.recognize_sface = val;
            else if (key == "cascade_path") cfg.cascade_path = val;
            else if (key == "detector") cfg.detector = val;
            else if (key == "image_format") cfg.image_format = val;
            else if (key == "frames") cfg.frames = std::stoi(val);
            else if (key == "debug") cfg.debug = (val == "yes");
            else if (key == "nogui") cfg.nogui = (val == "yes");
        }
        return true;
    }

    FA_EXPORT bool fa_capture_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& device_path, std::string& log) {
        cv::VideoCapture cap(device_path);
        if (!cap.isOpened()) { log = "Webcam non accessibile."; return false; }

        cv::Ptr<cv::FaceDetectorYN> detector_yn;
        cv::CascadeClassifier detector_cascade;

        if (cfg.detector == "yunet") detector_yn = cv::FaceDetectorYN::create(cfg.detect_yunet, "", cv::Size(320, 320));
        else if (cfg.detector == "cascade") detector_cascade.load(cfg.cascade_path);

        std::string user_dir = cfg.basedir + "/captures/" + user;
        fs::create_directories(user_dir);

        int start_idx = get_last_index(user_dir) + 1;
        int current_saved = 0;

        while (current_saved < cfg.frames) {
            cv::Mat frame; cap >> frame;
            if (frame.empty()) continue;

            bool face_found = (cfg.detector == "none");
            if (cfg.detector == "yunet") {
                detector_yn->setInputSize(frame.size());
                cv::Mat faces; detector_yn->detect(frame, faces);
                face_found = (faces.rows > 0);
            } else if (cfg.detector == "cascade") {
                cv::Mat gray; cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
                std::vector<cv::Rect> faces;
                detector_cascade.detectMultiScale(gray, faces, 1.1, 3);
                face_found = !faces.empty();
            }

            if (face_found) {
                cv::Mat res; cv::resize(frame, res, cv::Size(cfg.width, cfg.height));
                std::string img_path = user_dir + "/frame_" + std::to_string(start_idx + current_saved) + "." + cfg.image_format;
                cv::imwrite(img_path, res);
                if (cfg.debug) std::cout << "\n[DEBUG] Salvato: " << img_path << std::flush;
                current_saved++;
            }

            if (!cfg.debug) std::cout << "\r[*] Cattura (" << cfg.detector << "): " << current_saved << "/" << cfg.frames << std::flush;

            if (!cfg.nogui) {
                cv::imshow("Cattura", frame);
                if (cv::waitKey(1) == 27) break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds((int)(cfg.capture_delay * 1000)));
        }
        if (!cfg.nogui) cv::destroyAllWindows();
        std::cout << std::endl;
        return true;
    }

    FA_EXPORT bool fa_train_user(const std::string& user, const FacialAuthConfig& cfg, std::string& log) {
        std::string user_dir = cfg.basedir + "/captures/" + user;
        if (!fs::exists(user_dir)) { log = "Nessuna cattura trovata."; return false; }

        std::vector<cv::Mat> images;
        for (const auto& entry : fs::directory_iterator(user_dir)) {
            cv::Mat img = cv::imread(entry.path().string());
            if (!img.empty()) images.push_back(img);
        }
        if (images.empty()) { log = "Immagini non valide."; return false; }

        std::string model_path = fa_user_model_path(cfg, user);
        fs::create_directories(cfg.modeldir);
        cv::FileStorage fs_out(model_path, cv::FileStorage::WRITE);

        // Header richiesto
        fs_out << "header" << "{";
        fs_out << "user" << user << "method" << cfg.method << "timestamp" << (int)std::time(nullptr);
        fs_out << "}";

        if (cfg.method == "sface") {
            cv::Ptr<cv::FaceRecognizerSF> face_rec = cv::FaceRecognizerSF::create(cfg.recognize_sface, "");
            cv::Mat mean_feat;
            for (const auto& img : images) {
                cv::Mat feat; face_rec->feature(img, feat);
                if (mean_feat.empty()) mean_feat = feat.clone(); else mean_feat += feat;
            }
            mean_feat /= static_cast<float>(images.size());
            fs_out << "embedding" << mean_feat;
        } else {
            cv::Ptr<cv::face::FaceRecognizer> model;
            std::vector<cv::Mat> grays;
            for (auto& img : images) { cv::Mat g; cv::cvtColor(img, g, cv::COLOR_BGR2GRAY); grays.push_back(g); }
            std::vector<int> lbls(grays.size(), 1);

            if (cfg.method == "lbph") model = cv::face::LBPHFaceRecognizer::create();
            else if (cfg.method == "eigen") model = cv::face::EigenFaceRecognizer::create();
            else if (cfg.method == "fisher") model = cv::face::FisherFaceRecognizer::create();

            model->train(grays, lbls);
            model->write(fs_out);
        }
        fs_out.release();
        log = "Modello salvato: " + model_path;
        return true;
    }

    FA_EXPORT bool fa_clean_captures(const std::string& user, const FacialAuthConfig& cfg, std::string& log) {
        fs::remove_all(cfg.basedir + "/captures/" + user);
        return true;
    }

}
