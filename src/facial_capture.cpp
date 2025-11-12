#include "Utils.h"
#include <iostream>

int main(int argc, char **argv) {
    FacialAuthConfig cfg;
    // piccola lettura config e override cli
    read_kv_config("/etc/pam_facial_auth/pam_facial.conf", cfg, nullptr);
    for (int i=1;i<argc;i++){
        std::string a(argv[i]);
        if (a.rfind("device=",0)==0) cfg.device = a.substr(7);
        else if (a.rfind("width=",0)==0) cfg.width = std::stoi(a.substr(6));
        else if (a.rfind("height=",0)==0) cfg.height = std::stoi(a.substr(7));
        else if (a=="debug") cfg.debug = true;
        else if (a=="nogui") cfg.nogui = true;
    }

    std::string used;
    cv::VideoCapture cap;
    if (!open_camera(cfg, cap, used)) {
        std::cerr << "Cannot open camera\n";
        return 1;
    }
    std::cout << "Camera: " << used << " " << cfg.width << "x" << cfg.height << "\n";
    while (true) {
        cv::Mat f; cap >> f; if (f.empty()) continue;
        cv::imshow("capture", f);
        if (cv::waitKey(1)==27) break;
    }
    return 0;
}
