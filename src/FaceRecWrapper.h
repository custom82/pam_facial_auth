#ifndef FACERECWRAPPER_H
#define FACERECWRAPPER_H

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>  // OpenCV face module

class FaceRecWrapper {
public:
    FaceRecWrapper(const std::string& modelPath, const std::string& name, const std::string& model_type);
    void Train(const std::vector<cv::Mat>& images, const std::vector<int>& labels);
    void Recognize(cv::Mat& face);
    void Load(const std::string& modelFile);
    void Predict(cv::Mat& face, int& prediction, double& confidence);

private:
    cv::Ptr<cv::face::LBPHFaceRecognizer> recognizer;
    std::string modelType;
};

#endif // FACERECWRAPPER_H
