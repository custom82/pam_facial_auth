#ifndef FACERECWRAPPER_H
#define FACERECWRAPPER_H

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <vector>
#include <string>

class FaceRecWrapper
{
public:
    FaceRecWrapper();
    FaceRecWrapper(const std::string& techniqueName, const std::string& pathCascade);

    void Train(const std::vector<cv::Mat>& images, const std::vector<int>& labels);
    void Predict(const cv::Mat& im, int& label, double& confidence);
    void Save(const std::string& path);
    void Load(const std::string& path);
    void SetLabelNames(const std::vector<std::string>& names);
    std::string GetLabelName(int index);

private:
    bool SetTechnique(const std::string& t);
    bool CropFace(const cv::Mat& image, cv::Mat& cropped);
    bool LoadCascade(const std::string& filepath);

    cv::Ptr<cv::face::FaceRecognizer> fr;
    cv::CascadeClassifier cascade;
    std::size_t sizeFace;
    std::string technique;
    std::string pathCascade;
};

#endif // FACERECWRAPPER_H

