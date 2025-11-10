#include "FaceRecWrapper.h"
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <iostream>

FaceRecWrapper::FaceRecWrapper(const std::string &modelPath, const std::string &name)
: modelPath(modelPath), name(name) {
	// Crea un FaceRecognizer
	fr = cv::face::EigenFaceRecognizer::create();
}

void FaceRecWrapper::Load(const std::string &path) {
	// Carica il modello addestrato
	fr->read(modelPath);
}

int FaceRecWrapper::Predict(const cv::Mat &image, int &prediction, double &confidence) {
	// La funzione predict non restituisce un valore, rimuovi il return
	fr->predict(image, prediction, confidence);  // Predizione dell'immagine
	return 0;  // Restituisci un valore, ad esempio 0 per successo
}
