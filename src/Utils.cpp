#include "Utils.h"
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

// Funzione per leggere la configurazione da un file
bool Utils::GetConfig(const std::string& configPath, std::map<std::string, std::string>& config) {
    std::ifstream file(configPath);
    if (!file.is_open()) {
        std::cerr << "Errore nell'aprire il file di configurazione: " << configPath << std::endl;
        return false;
    }

    std::string key, value;
    while (file >> key >> value) {
        config[key] = value;
    }

    file.close();
    return true;
}

// Funzione per caricare un'immagine e convertirla in scala di grigi
cv::Mat Utils::LoadImage(const std::string& imagePath) {
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Errore nel caricare l'immagine: " << imagePath << std::endl;
    }
    return image;
}

// Funzione per ridimensionare un'immagine
cv::Mat Utils::ResizeImage(const cv::Mat& image, int width, int height) {
    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size(width, height));
    return resizedImage;
}

// Funzione per visualizzare un'immagine
void Utils::ShowImage(const cv::Mat& image, const std::string& windowName) {
    cv::imshow(windowName, image);
    cv::waitKey(0); // Aspetta una chiave per chiudere la finestra
}

// Funzione di esempio per fare qualcosa con l'immagine
void Utils::ProcessImage(const cv::Mat& image) {
    // Fai qualcosa con l'immagine, per esempio, creare un histogramma
    cv::Mat hist;
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    cv::calcHist(&image, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
    
    // Visualizza il risultato
    std::cout << "Lunghezza dell'istogramma: " << hist.rows << std::endl;
}

