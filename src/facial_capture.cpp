#include "libfacialauth.h"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <filesystem>

namespace fs = std::filesystem;

// Funzione helper per creare la directory se non esiste
std::string fa_user_image_dir(const FacialAuthConfig &cfg, const std::string &user) {
    std::string path = cfg.basedir + "/" + user + "/captures";
    fs::create_directories(path);
    return path;
}

void debug_dump(const FacialAuthConfig& cfg) {
    std::cout << "--- Config Dump ---\n"
    << " device=" << cfg.device << "\n"
    << " method=" << cfg.training_method << "\n"
    << " debug=" << (cfg.debug ? "yes" : "no") << "\n"
    << " ignore=" << (cfg.ignore_failure ? "yes" : "no") << "\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: facial_capture <user>\n";
        return 1;
    }
    std::string user = argv[1];

    FacialAuthConfig cfg;
    std::string log;
    if (!fa_load_config(cfg, log, FACIALAUTH_DEFAULT_CONFIG)) {
        std::cerr << "[WARN] Config non caricata, uso default: " << log << "\n";
    }

    if (cfg.debug) debug_dump(cfg);

    std::string capture_path = fa_user_image_dir(cfg, user);
    std::cout << "[INFO] Acquisizione per " << user << " in: " << capture_path << "\n";

    cv::VideoCapture cap;
    if (cfg.device.find_first_not_of("0123456789") == std::string::npos)
        cap.open(std::stoi(cfg.device));
    else
        cap.open(cfg.device);

    if (!cap.isOpened()) {
        std::cerr << "[ERROR] Impossibile aprire il dispositivo: " << cfg.device << "\n";
        return 1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, cfg.width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.height);

    int count = 0;
    while (count < cfg.frames) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        std::string filename = capture_path + "/img_" + std::to_string(count) + "." + cfg.image_format;
        cv::imwrite(filename, frame);

        std::cout << "\rCattura frame " << count + 1 << "/" << cfg.frames << std::flush;
        count++;
        cv::waitKey(cfg.sleep_ms);
    }

    std::cout << "\n[SUCCESS] Cattura completata.\n";
    return 0;
}
