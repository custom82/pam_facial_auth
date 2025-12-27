#include "../include/libfacialauth.h"
#include <iostream>
#include <filesystem>
#include <getopt.h>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp> // Per Haar e YuNet
#include <opencv2/dnn.hpp>
#include <thread>

namespace fs = std::filesystem;

// Funzione helper per il rilevamento del volto
bool detect_face(cv::Mat& frame, const std::string& method, cv::Ptr<cv::CascadeClassifier>& haar, cv::Ptr<cv::FaceDetectorYN>& yunet) {
    if (method == "haar") {
        std::vector<cv::Rect> faces;
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        haar->detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(30, 30));
        return !faces.empty();
    } else if (method == "yunet") {
        cv::Mat faces;
        yunet->setInputSize(frame.size());
        yunet->detect(frame, faces);
        return faces.rows > 0;
    }
    return true; // Se "none", accettiamo tutto
}

int main(int argc, char** argv) {
    FacialAuthConfig cfg;
    std::string user, detector_type = "none";
    std::string config_path = FACIALAUTH_DEFAULT_CONFIG;
    bool force = false, flush = false, nogui = false, verbose = false;

    static struct option long_options[] = {
        {"user",       required_argument, 0, 'u'},
        {"detector",   required_argument, 0, 'D'}, // Nuova opzione
        {"config",     required_argument, 0, 'c'},
        {"device",     required_argument, 0, 'd'},
        {"force",      no_argument,       0, 'f'},
        {"flush",      no_argument,       0, 'C'},
        {"num_images", required_argument, 0, 'n'},
        {"sleep",      required_argument, 0, 's'},
        {"nogui",      no_argument,       0, 'G'},
        {"help",       no_argument,       0, 'H'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "u:D:c:d:fn:s:GH", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'u': user = optarg; break;
            case 'D': detector_type = optarg; break;
            case 'c': config_path = optarg; break;
            case 'd': cfg.device = optarg; break;
            case 'f': force = true; break;
            case 'C': flush = true; break;
            case 'n': cfg.frames = std::stoi(optarg); break;
            case 's': cfg.sleep_ms = static_cast<int>(std::stod(optarg) * 1000); break;
            case 'G': nogui = true; break;
            case 'H': /* print help */ return 0;
        }
    }

    // Inizializzazione Detector
    cv::Ptr<cv::CascadeClassifier> haar_detector;
    cv::Ptr<cv::FaceDetectorYN> yunet_detector;

    if (detector_type == "haar") {
        haar_detector = cv::makePtr<cv::CascadeClassifier>("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml");
    } else if (detector_type == "yunet") {
        // YuNet richiede il file face_detection_yunet_2023mar.onnx
        yunet_detector = cv::FaceDetectorYN::create(cfg.detect_model_path, "", cv::Size(320, 320));
    }

    // [ ... Logica di apertura camera e directory come prima ... ]

    int count = 0;
    while (count < cfg.frames) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        // Eseguiamo la cattura solo se il volto è rilevato (o se detector è none)
        if (detect_face(frame, detector_type, haar_detector, yunet_detector)) {
            std::string filename = user_dir + "/img_" + std::to_string(count) + "." + cfg.image_format;
            cv::imwrite(filename, frame);
            count++;
            if (verbose) std::cout << "Volto rilevato e salvato (" << count << "/" << cfg.frames << ")\n";
        } else if (verbose) {
            std::cout << "Nessun volto rilevato, salto il frame...\n";
        }

        if (!nogui) {
            cv::imshow("Cattura", frame);
            if (cv::waitKey(1) == 'q') break;
        }

        if (cfg.sleep_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(cfg.sleep_ms));
    }

    return 0;
}
