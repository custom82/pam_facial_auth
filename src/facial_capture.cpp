#include "libfacialauth.h"
#include <iostream>

int facial_capture_main(int argc, char **argv);

int main(int argc, char **argv)
{
    try {
        return facial_capture_main(argc, argv);
    }
    catch (const cv::Exception &e) {
        std::cerr << "[OpenCV ERROR] " << e.what() << std::endl;
        return 1;
    }
    catch (const std::exception &e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 1;
    }
}
