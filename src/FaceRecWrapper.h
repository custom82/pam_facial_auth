#ifndef FACERECWRAPPER_H
#define FACERECWRAPPER_H

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <vector>
#include <string>

class FaceRecWrapper {
public:
    // Costruttore con i parametri di percorso del modello e nome
    FaceRecWrapper(const std::string &modelPath, const std::string &name);

    // Carica il modello
    void Load(const std::string &path);

    // Allena il riconoscitore con immagini e etichette
    void Train(const std::vector<cv::Mat> &images, const std::vector<int> &labels);

    // Predice l'etichetta di una faccia nell'immagine
    int Predict(const cv::Mat &image, int &prediction, double &confidence);

    // Aggiungi i nomi delle etichette
    void SetLabelNames(const std::vector<std::string> &names);

    // Ottieni il nome dell'etichetta per l'indice
    std::string GetLabelName(int index);

private:
    // Riconoscitore facciale
    cv::Ptr<cv::face::FaceRecognizer> fr;

    // Vettore per memorizzare i nomi delle etichette
    std::vector<std::string> labelNames;

    // Percorso del modello
    std::string modelPath;
};

#endif // FACERECWRAPPER_H
