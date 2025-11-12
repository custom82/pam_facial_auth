#include <iostream>
#include <opencv2/opencv.hpp>
#include "FaceRecWrapper.h"

int main(int argc, char** argv)
{
    // Ensure a model file and name are passed
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <path_to_model> <username>" << std::endl;
        return -1;
    }

    // Initialize FaceRecWrapper
    FaceRecWrapper faceRec(argv[1], argv[2], "LBPH");  // Assuming model type is LBPH

    // Start capturing face
    cv::VideoCapture capture(0);
    if (!capture.isOpened()) {
        std::cerr << "Error: Could not open webcam." << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (capture.read(frame)) {
        // Process the frame for face recognition here
        faceRec.Recognize(frame);
        cv::imshow("Facial Authentication", frame);

        if (cv::waitKey(1) == 27)  // Press ESC to quit
            break;
    }

    return 0;
}
