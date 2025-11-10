#include "FaceRecWrapper.h"
#include <opencv2/face.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

FaceRecWrapper::FaceRecWrapper(const std::string &modelPath, const std::string &name)
: modelPath(modelPath), name(name) {
	// Inizializzazione
	fr = cv::face::EigenFaceRecognizer::create();
}

void FaceRecWrapper::Load(const std::string &path) {
	// Caricamento del modello
	fr->read(path);
}

int FaceRecWrapper::Predict(const cv::Mat &image, int &prediction, double &confidence) {
	// Predizione della faccia
	cv::Mat gray;
	if (image.channels() == 3) {
		cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
	} else {
		gray = image;
	}
	return fr->predict(gray, prediction, confidence);
}
