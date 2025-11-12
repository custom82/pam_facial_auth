#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include "FaceRecWrapper.h"
#include "Utils.h"
#include <iostream>

int main(int argc, char** argv) {
    cv::VideoCapture cap(0);  // apri la webcam

    if (!cap.isOpened()) {
        std::cerr << "Errore nell'aprire la webcam!" << std::endl;
        return -1;
    }

    int width = 1280;
    int height = 720;
    cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);

    FaceRecWrapper faceRec("path/to/model.xml", "root");

    cv::Mat frame;
    while (true) {
        cap >> frame;  // acquisisci frame
        if (frame.empty()) {
            break;
        }

        int prediction = -1;
        double confidence = 0;
        faceRec.Predict(frame, prediction, confidence);

        cv::imshow("Facial Capture", frame);
        if (cv::waitKey(1) == 27) break; // premere ESC per uscire
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
