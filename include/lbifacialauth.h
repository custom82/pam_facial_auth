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

#ifndef FACIAL_AUTH_H
#define FACIAL_AUTH_H

#include <string>
#include <opencv2/opencv.hpp>

class FacialAuth {
public:
    FacialAuth();
    ~FacialAuth();

    bool Authenticate(const std::string &user);  // Metodo per autenticare un utente

private:
    bool LoadModel(const std::string &modelPath); // Carica il modello di riconoscimento facciale
    bool TrainModel(const std::vector<cv::Mat> &images, const std::vector<int> &labels);  // Metodo per addestrare il modello (opzionale)
    bool RecognizeFace(const cv::Mat &faceImage);  // Riconosce il volto in un'immagine

    cv::Ptr<cv::face::LBPHFaceRecognizer> recognizer;  // Riconoscitore facciale OpenCV
    std::string modelPath;  // Percorso del modello facciale
};

#endif // FACIAL_AUTH_H
