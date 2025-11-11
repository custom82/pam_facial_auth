#ifndef FACERECWRAPPER_H
#define FACERECWRAPPER_H

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <vector>
#include <string>

class FaceRecWrapper {
public:
    // Modifica il costruttore per includere il parametro model
    FaceRecWrapper(const std::string &modelPath, const std::string &name, const std::string &model = "eigenfaces");

    void Load(const std::string &path);
    int Predict(const cv::Mat &image, int &prediction, double &confidence);

private:
    cv::Ptr<cv::face::FaceRecognizer> fr; // Oggetto riconoscitore facciale
    std::string modelPath;                // Percorso del modello
    std::string name;                     // Nome dell'utente
    std::string model;                    // Tipo di modello (e.g., eigenfaces, lbph)
};

#endif // FACERECWRAPPER_H

