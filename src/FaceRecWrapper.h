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
    void Train(const std::vector<cv::Mat> &images, const std::vector<int> &labels);
    int Predict(const cv::Mat &image, int &prediction, double &confidence);

    // Aggiunta delle funzioni per la gestione delle etichette
    void SetLabelNames(const std::vector<std::string> &names);
    std::string GetLabelName(int index);

private:
    cv::Ptr<cv::face::FaceRecognizer> fr;
    std::vector<std::string> labelNames;  // Aggiunto per memorizzare le etichette
    std::string modelPath;
};

#endif // FACERECWRAPPER_H
