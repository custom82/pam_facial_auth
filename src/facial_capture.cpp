#include "libfacialauth.h"
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "Usage: facial_capture <user>\n"; return 1; }
    std::string user = argv[1];

    FacialAuthConfig cfg;
    std::string log;
    fa_load_config(cfg, log, FACIALAUTH_DEFAULT_CONFIG);

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) return 1;

    std::string path = cfg.basedir + "/" + user + "/captures";
    fs::create_directories(path);

    for (int i = 0; i < cfg.frames; ++i) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;
        cv::imwrite(path + "/img_" + std::to_string(i) + ".jpg", frame);
        cv::imshow("Cattura", frame);
        cv::waitKey(100);
    }
    return 0;
}
