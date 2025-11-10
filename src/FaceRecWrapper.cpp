#include "FaceRecWrapper.h"
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>

// Costruttore per inizializzare il percorso del modello
FaceRecWrapper::FaceRecWrapper(const std::string &modelPath, const std::string &name)
: modelPath(modelPath) {
	// Se il modello esiste, carica il riconoscitore facciale
	Load(modelPath);
}

// Funzione per caricare il modello
void FaceRecWrapper::Load(const std::string &path) {
	// Inizializza il riconoscitore facciale con la versione corretta per OpenCV 4.x
	fr = cv::face::EigenFaceRecognizer::create();  // Usa EigenFaces come esempio
	fr->read(path + "-facerec.xml"); // Carica il modello da un file XML
	std::cout << "Model loaded from: " << path << std::endl;
}

// Funzione per allenare il riconoscitore facciale con immagini e etichette
void FaceRecWrapper::Train(const std::vector<cv::Mat> &images, const std::vector<int> &labels) {
	if (images.empty()) {
		std::cerr << "Error: No images provided for training!" << std::endl;
		return;
	}
	fr->train(images, labels); // Allena il riconoscitore
}

// Funzione per fare la predizione dell'etichetta di una faccia in un'immagine
int FaceRecWrapper::Predict(const cv::Mat &image, int &prediction, double &confidence) {
	fr->predict(image, prediction, confidence); // Predice l'etichetta
	return prediction;  // Restituisce il valore di predizione
}

// Funzione per impostare i nomi delle etichette
void FaceRecWrapper::SetLabelNames(const std::vector<std::string> &names) {
	labelNames = names;
	// Aggiungi i nomi delle etichette al riconoscitore
	for (int i = 0; i < labelNames.size(); ++i) {
		fr->setLabelInfo(i, labelNames[i]);
	}
}

// Funzione per ottenere il nome dell'etichetta in base all'indice
std::string FaceRecWrapper::GetLabelName(int index) {
	if (index >= 0 && index < labelNames.size()) {
		return labelNames[index];
	}
	return "Unknown";  // Se l'indice non è valido
}

