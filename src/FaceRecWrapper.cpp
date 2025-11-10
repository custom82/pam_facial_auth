#include "FaceRecWrapper.h"
#include <opencv2/face.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

FaceRecWrapper::FaceRecWrapper(const std::string &modelPath, const std::string &name)
: modelPath(modelPath) {
	// Se il modello esiste, carica il riconoscitore facciale
	Load(modelPath);
}

// Carica il modello e la tecnica
void FaceRecWrapper::Load(const std::string &path) {
	// Carica il riconoscitore facciale (modifica in base alla tecnica scelta)
	fr = cv::face::createEigenFaceRecognizer(); // Per esempio, usa EigenFaces
	fr->read(path + "-facerec.xml");
	// Altre operazioni di caricamento del modello
	std::cout << "Model loaded from: " << path << std::endl;
}

// Allena il riconoscitore facciale con immagini e etichette
void FaceRecWrapper::Train(const std::vector<cv::Mat> &images, const std::vector<int> &labels) {
	if (images.empty()) {
		std::cerr << "Error: No images provided for training!" << std::endl;
		return;
	}
	fr->train(images, labels);
}

// Predice l'etichetta di una faccia in un'immagine
int FaceRecWrapper::Predict(const cv::Mat &image, int &prediction, double &confidence) {
	return fr->predict(image, prediction, confidence);
}

// Imposta i nomi delle etichette
void FaceRecWrapper::SetLabelNames(const std::vector<std::string> &names) {
	labelNames = names;
	// Aggiungi i nomi delle etichette al riconoscitore
	for (int i = 0; i < labelNames.size(); ++i) {
		fr->setLabelInfo(i, labelNames[i]);
	}
}

// Restituisce il nome dell'etichetta per un dato indice
std::string FaceRecWrapper::GetLabelName(int index) {
	if (index >= 0 && index < labelNames.size()) {
		return labelNames[index];
	}
	return "Unknown";  // Se l'indice non è valido
}
