#include "FaceRecWrapper.h"
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <fstream>
#include <iostream>

FaceRecWrapper::FaceRecWrapper(const std::string &modelPath, const std::string &name) {
	// Inizializza il riconoscitore con un tipo predefinito
	fr = cv::face::EigenFaceRecognizer::create();
	Load(modelPath);  // Carica il modello dal percorso
}

void FaceRecWrapper::Load(const std::string &path) {
	fr->read(path);  // Carica il modello dal file
}

void FaceRecWrapper::Train(const std::vector<cv::Mat> &images, const std::vector<int> &labels) {
	fr->train(images, labels);
}

int FaceRecWrapper::Predict(const cv::Mat &image, int &prediction, double &confidence) {
	fr->predict(image, prediction, confidence);  // Usa il riconoscitore per fare una previsione
	return prediction;
}

void FaceRecWrapper::SetLabelNames(const std::vector<std::string> &names) {
	labelNames = names;
}

std::string FaceRecWrapper::GetLabelName(int index) {
	if (index < 0 || index >= labelNames.size()) {
		return "";
	}
	return labelNames[index];
}
