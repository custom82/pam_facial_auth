#include "../include/libfacialauth.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <thread>
#include <chrono>
#include <unistd.h>

namespace fs = std::filesystem;

// Verifica se l'utente ha i privilegi di root
bool fa_check_root(const std::string& tool_name) {
    if (geteuid() != 0) {
        std::cerr << "[ERROR] " << tool_name << " deve essere eseguito come root.\n";
        return false;
    }
    return true;
}

// Verifica l'esistenza di un file
bool fa_file_exists(const std::string& path) {
    return fs::exists(path);
}

// Cattura un dataset di immagini per l'addestramento
bool fa_capture_dataset(const FacialAuthConfig& cfg, std::string& log, const std::string& user, int count) {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        log = "Impossibile aprire la fotocamera.";
        return false;
    }

    std::string user_dir = "/var/lib/pam_facial_auth/data/" + user;
    if (!fs::exists(user_dir)) {
        fs::create_directories(user_dir);
    }

    cv::CascadeClassifier face_cascade;
    if (!face_cascade.load(cfg.cascade_path)) {
        log = "Impossibile caricare il classificatore Haar.";
        return false;
    }

    int saved = 0;
    cv::Mat frame, gray;

    while (saved < count) {
        cap >> frame;
        if (frame.empty()) break;

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        std::vector<cv::Rect> faces;
        face_cascade.detectMultiScale(gray, faces, 1.1, 4);

        for (const auto& area : faces) {
            cv::Mat face_roi = gray(area);
            std::string filename = user_dir + "/" + std::to_string(saved) + "." + cfg.image_format;
            cv::imwrite(filename, face_roi);
            saved++;

            if (cfg.debug) {
                std::cout << "[DEBUG] Salvata immagine " << saved << "/" << count << std::endl;
            }
            if (saved >= count) break;
        }

        if (cfg.sleep_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(cfg.sleep_ms));
        }
    }

    log = "Cattura completata: " + std::to_string(saved) + " immagini salvate.";
    return true;
}

// Wrapper per la cattura dell'utente
bool fa_capture_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& detector, std::string& log) {
    return fa_capture_dataset(cfg, log, user, 50); // Default 50 immagini
}

// Addestra il modello per un utente specifico
bool fa_train_user(const std::string& user, const FacialAuthConfig& cfg, std::string& log) {
    std::string user_dir = "/var/lib/pam_facial_auth/data/" + user;
    std::string model_path = "/var/lib/pam_facial_auth/models/" + user + ".xml";

    if (!fs::exists(user_dir)) {
        log = "Dataset non trovato per l'utente " + user;
        return false;
    }

    std::vector<cv::Mat> images;
    std::vector<int> labels;

    for (const auto& entry : fs::directory_iterator(user_dir)) {
        images.push_back(cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE));
        labels.push_back(0); // Etichetta singola per l'utente
    }

    if (images.empty()) {
        log = "Nessuna immagine trovata per il training.";
        return false;
    }

    cv::Ptr<cv::face::FaceRecognizer> model;
    if (cfg.model_type == "lbph") {
        model = cv::face::LBPHFaceRecognizer::create();
    } else {
        log = "Tipo di modello non supportato: " + cfg.model_type;
        return false;
    }

    model->train(images, labels);

    if (!fs::exists("/var/lib/pam_facial_auth/models")) {
        fs::create_directories("/var/lib/pam_facial_auth/models");
    }

    model->save(model_path);
    log = "Modello salvato in: " + model_path;
    return true;
}

// Esegue il test di riconoscimento
bool fa_test_user(const std::string& user, const FacialAuthConfig& cfg, std::string& log) {
    std::string model_path = "/var/lib/pam_facial_auth/models/" + user + ".xml";
    if (!fa_file_exists(model_path)) {
        log = "Modello non trovato per " + user;
        return false;
    }

    cv::Ptr<cv::face::FaceRecognizer> model = cv::face::LBPHFaceRecognizer::create();
    model->read(model_path);

    cv::VideoCapture cap(0);
    cv::Mat frame, gray;
    cap >> frame;
    if (frame.empty()) {
        log = "Errore cattura test.";
        return false;
    }

    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    int label = -1;
    double confidence = 0.0;
    model->predict(gray, label, confidence);

    if (label == 0 && confidence < cfg.threshold) {
        log = "Riconoscimento riuscito (Conf: " + std::to_string(confidence) + ")";
        return true;
    }

    log = "Riconoscimento fallito.";
    return false;
}
