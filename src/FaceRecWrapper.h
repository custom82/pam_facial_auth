#ifndef FACERECWRAPPER_H
#define FACERECWRAPPER_H

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <string>

class FaceRecWrapper
{
public:
    // Constructor with model path, name, and model type
    FaceRecWrapper(const std::string& modelPath, const std::string& name, const std::string& model_type);

    // Load the model
    void Load(const std::string& modelPath);

    // Predict a given image
    void Predict(const cv::Mat& image, int& predicted_label, double& confidence);

    // Recognize a face in a frame
    void Recognize(cv::Mat& frame);

    // Train the recognizer
    void Train(const std::vector<cv::Mat>& images, const std::vector<int>& labels);

private:
    cv::Ptr<cv::face::LBPHFaceRecognizer> recognizer;
    std::string modelPath;
    std::string model_type;
};

#endif
