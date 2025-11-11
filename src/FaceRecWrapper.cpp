#include "FaceRecWrapper.h"
#include <iostream>

FaceRecWrapper::FaceRecWrapper(const std::string &modelPath, const std::string &name)
: modelPath(modelPath), name(name) {
	// Inizializzazione del riconoscitore
	if (modelPath.empty()) {
		std::cerr << "Model path is empty!" << std::endl;
		return;
	}

	// Carica il modello in base al tipo configurato
	if (name == "lbph") {
		fr = cv::face::LBPHFaceRecognizer::create();
	} else if (name == "eigenfaces") {
		fr = cv::face::EigenFaceRecognizer::create();
	} else if (name == "fisherfaces") {
		fr = cv::face::FisherFaceRecognizer::create();
	}

	// Carica il modello dal percorso
	if (!modelPath.empty() && fs::exists(modelPath)) {
		fr->read(modelPath);  // Carica il modello
	} else {
		std::cerr << "Model file not found: " << modelPath << std::endl;
	}
}

void FaceRecWrapper::Load(const std::string &path) {
	if (fs::exists(path)) {
		fr->read(path);  // Carica il modello
	}
}

int FaceRecWrapper::Predict(const cv::Mat &image, int &prediction, double &confidence) {
	// Predizione usando il riconoscitore facciale
	fr->predict(image, prediction, confidence);
	return prediction;  // Return prediction for further use (not necessary, but useful if needed)
}
