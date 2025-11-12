#include "../include/FaceRecWrapper.h"
#include "../include/FacialAuth.h"
#include <opencv2/face.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <iostream>

FaceRecWrapper::FaceRecWrapper(const std::string& modelPath, const std::string& name, const std::string& model_type)
: modelType(model_type) {
	recognizer = cv::face::LBPHFaceRecognizer::create();  // Inizializza il riconoscitore
}

void FaceRecWrapper::Train(const std::vector<cv::Mat>& images, const std::vector<int>& labels) {
	recognizer->train(images, labels);
}

void FaceRecWrapper::Recognize(cv::Mat& face) {
	int label = -1;
	double confidence = 0.0;
	recognizer->predict(face, label, confidence);
}

void FaceRecWrapper::Load(const std::string& modelFile) {
	recognizer->read(modelFile);  // Carica il modello dal file
}

void FaceRecWrapper::Predict(cv::Mat& face, int& prediction, double& confidence) {
	recognizer->predict(face, prediction, confidence);
}

FacialAuth::FacialAuth() {
	recognizer = cv::face::LBPHFaceRecognizer::create();
}

FacialAuth::~FacialAuth() {
	// Distruttore: eventuale rilascio risorse
}

bool FacialAuth::Authenticate(const std::string &user) {
	cv::Mat faceImage;
	faceImage = cv::imread("path_to_user_image.jpg", cv::IMREAD_GRAYSCALE);

	if (faceImage.empty()) {
		std::cerr << "Errore: Immagine del volto non trovata!" << std::endl;
		return false;
	}

	if (!recognizer->isTrained()) {
		if (!LoadModel("path_to_model.xml")) {
			std::cerr << "Errore: Impossibile caricare il modello facciale!" << std::endl;
			return false;
		}
	}

	return RecognizeFace(faceImage);
}

bool FacialAuth::LoadModel(const std::string &modelPath) {
	try {
		recognizer->read(modelPath);
		this->modelPath = modelPath;
		return true;
	} catch (const std::exception &e) {
		std::cerr << "Errore durante il caricamento del modello: " << e.what() << std::endl;
		return false;
	}
}

bool FacialAuth::RecognizeFace(const cv::Mat &faceImage) {
	int label = -1;
	double confidence = 0.0;
	recognizer->predict(faceImage, label, confidence);

	if (confidence < 50) {  // Soglia di confidenza
		std::cout << "Autenticazione riuscita! Utente: " << label << std::endl;
		return true;
	} else {
		std::cout << "Autenticazione fallita. Confidenza: " << confidence << std::endl;
		return false;
	}
}

