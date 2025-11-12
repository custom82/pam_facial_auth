#ifndef FACERECWRAPPER_H
#define FACERECWRAPPER_H

#include <opencv2/opencv.hpp>
#include <string>

class FaceRecWrapper
{
public:
    // Updated constructor with 3 parameters
    FaceRecWrapper(const std::string& modelPath, const std::string& name, const std::string& model_type);

    // Method to recognize faces
    void Recognize(cv::Mat& frame);

    // Method to train the model
    void Train(const std::vector<cv::Mat>& images, const std::vector<int>& labels);

private:
    cv::Ptr<cv::face::LBPHFaceRecognizer> recognizer;
    std::string modelPath;
    std::string model_type;
};

#endif
