#include "FaceRecWrapper.h"
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>

FaceRecWrapper::FaceRecWrapper(const std::string &modelPath, const std::string &name, const std::string &model)
: modelPath(modelPath), name(name), model(model) {

	// Seleziona il tipo di riconoscitore in base al parametro 'model'
	if (model == "lbph") {
		fr = cv::face::LBPHFaceRecognizer::create();
		std::cout << "Using LBPH model" << std::endl;
	} else if (model == "eigenfaces") {
		fr = cv::face::EigenFaceRecognizer::create();
		std::cout << "Using Eigenfaces model" << std::endl;
	} else if (model == "fisherfaces") {
		fr = cv::face::FisherFaceRecognizer::create();
		std::cout << "Using Fisherfaces model" << std::endl;
	} else {
		std::cerr << "Invalid model type: " << model << ", defaulting to Eigenfaces" << std::endl;
		fr = cv::face::EigenFaceRecognizer::create();
	}
}

void FaceRecWrapper::Load(const std::string &path) {
	// Carica il modello addestrato
	fr->read(path);
}

int FaceRecWrapper::Predict(const cv::Mat &image, int &prediction, double &confidence) {
	// Esegui la previsione sul frame
	return fr->predict(image, prediction, confidence);
}
