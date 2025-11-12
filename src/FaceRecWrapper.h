#ifndef FACERECWRAPPER_H
#define FACERECWRAPPER_H

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <string>

class FaceRecWrapper {
public:
    // model_type: "lbph" | "eigen" | "fisher"
    FaceRecWrapper(const std::string &modelPath, const std::string &name, const std::string &model_type);
    void Load(const std::string &path); // carica file modello (xml/yaml)
    // ritorna 0=OK, !=0 errore; prediction e confidence valorizzati
    int Predict(const cv::Mat &image, int &prediction, double &confidence);
    // Salva nei formati richiesti (xml/yaml)
    bool SaveAll(const std::string &basePath, bool save_xml, bool save_yaml);

private:
    cv::Ptr<cv::face::FaceRecognizer> fr;
    std::string modelPath;
    std::string name;
    std::string type; // lbph/eigen/fisher
};

#endif
