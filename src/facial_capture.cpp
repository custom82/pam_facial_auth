#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>  // Ensure the OpenCV face module is included

#include "../include/FaceRecWrapper.h"  // Include the FaceRecWrapper class header

int main(int argc, char **argv) {
    // Check if the model path is provided
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path>" << std::endl;
        return -1;
    }

    // Create a FaceRecWrapper instance with 3 arguments (model path, name, and model type)
    FaceRecWrapper faceRec("path/to/model.xml", "root", "LBPH");

    // Capture face from the webcam
    cv::Mat frame;
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Could not open camera." << std::endl;
        return -1;
    }

    // Capture and process the face
    cap >> frame;
    if (!frame.empty()) {
        faceRec.Recognize(frame);  // Recognize faces using the recognizer
    }

    return 0;
}
