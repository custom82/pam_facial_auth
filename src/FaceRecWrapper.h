#ifndef FACERECWRAPPER_H
#define FACERECWRAPPER_H

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <string>
#include <vector>

class FaceRecWrapper {
public:
    FaceRecWrapper();
    FaceRecWrapper(const std::string& techniqueName, const std::string& pathCascade);

    void Train(const std::vector<cv::Mat>& images, const std::vector<int>& labels);
    void Predict(const cv::Mat& im, int& label, double& confidence);
    void Save(const std::string& path);
    void Load(const std::string& path);

private:
    bool SetTechnique(const std::string& t);
    bool LoadCascade(const std::string& filepath);
    bool CropFace(const cv::Mat& image, cv::Mat& cropped);

    int sizeFace;
    cv::Ptr<cv::face::FaceRecognizer> fr;
    cv::CascadeClassifier cascade;
    std::string pathCascade;
    std::string technique;
};

#endif // FACERECWRAPPER_H
