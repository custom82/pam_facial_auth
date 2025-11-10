#ifndef FACERECWRAPPER_H
#define FACERECWRAPPER_H

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <vector>
#include <string>

class FaceRecWrapper {
public:
    FaceRecWrapper(const std::string &modelPath, const std::string &name);
    void Load(const std::string &path);
    int Predict(const cv::Mat &image, int &prediction, double &confidence);

private:
    cv::Ptr<cv::face::FaceRecognizer> fr;
    std::string modelPath;
    std::string name;
};

#endif // FACERECWRAPPER_H
