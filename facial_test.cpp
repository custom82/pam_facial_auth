#include "FaceRecWrapper.h"
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
	// Inizializza il riconoscimento facciale
	std::string modelPath = "path/to/trained_model.yml";
	std::string name = "Face Recognizer";

	FaceRecWrapper frw(modelPath, name);

	// Carica il modello
	frw.Load(modelPath);

	// Carica un'immagine da testare
	cv::Mat image = cv::imread("path/to/image.jpg");
	if (image.empty()) {
		std::cerr << "Immagine non trovata!" << std::endl;
		return -1;
	}

	// Predizione
	int prediction;
	double confidence;
	frw.Predict(image, prediction, confidence);

	std::cout << "Predizione: " << prediction << ", Confidenza: " << confidence << std::endl;

	return 0;
}
