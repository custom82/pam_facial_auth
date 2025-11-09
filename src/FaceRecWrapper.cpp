#include "FaceRecWrapper.h"
#include <opencv2/face.hpp>
#include <opencv2/opencv.hpp>

FaceRecWrapper::FaceRecWrapper(const std::string &modelPath, const std::string &name)
: modelPath(modelPath) {
	fr = cv::face::LBPHFaceRecognizer::create();  // Usa LBPH o altro riconoscitore a seconda delle necessità
}

void FaceRecWrapper::Load(const std::string &path) {
	// Convert std::string to cv::String explicitly
	fr->read(cv::String(path + "-facerec.xml"));
}

void FaceRecWrapper::Train(const std::vector<cv::Mat> &images, const std::vector<int> &labels) {
	fr->train(images, labels);
}

int FaceRecWrapper::Predict(const cv::Mat &image, int &prediction, double &confidence) {
	return fr->predict(image, prediction, confidence);
}

// Funzione per impostare le etichette
void FaceRecWrapper::SetLabelNames(const std::vector<std::string> &names) {
	labelNames = names;
}

// Funzione per ottenere il nome dell'etichetta in base all'indice
std::string FaceRecWrapper::GetLabelName(int index) {
	if (index >= 0 && index < labelNames.size()) {
		return labelNames[index];
	}
	return "Unknown";  // Ritorna un valore di default se l'indice è invalido
}
