#include "../include/libfacialauth.h"
#include <iostream>
#include <filesystem>
#include <opencv2/highgui.hpp>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Uso: " << argv[0] << " <username>\n";
        return 1;
    }

    std::string user = argv[1];
    FacialAuthConfig cfg;
    std::string log;
    fa_load_config(cfg, log, FACIALAUTH_DEFAULT_CONFIG);

    std::string user_dir = cfg.basedir + "/" + user + "/captures";
    fs::create_directories(user_dir);

    cv::VideoCapture cap;
    try { cap.open(std::stoi(cfg.device)); } catch(...) { cap.open(cfg.device); }

    if (!cap.isOpened()) {
        std::cerr << "Errore: Impossibile aprire la camera " << cfg.device << "\n";
        return 1;
    }

    std::cout << "Cattura di " << cfg.frames << " frame per l'utente " << user << "...\n";
    std::cout << "Premi 'q' per interrompere.\n";

    int count = 0;
    while (count < cfg.frames) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        cv::imshow("Cattura Volto - " + user, frame);

        std::string filename = user_dir + "/img_" + std::to_string(count) + "." + cfg.image_format;
        cv::imwrite(filename, frame);

        count++;
        if (cv::waitKey(cfg.sleep_ms) == 'q') break;
    }

    std::cout << "Cattura completata in: " << user_dir << "\n";
    return 0;
}
