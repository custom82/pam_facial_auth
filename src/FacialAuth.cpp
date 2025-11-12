#include "FacialAuth.h"
#include <opencv2/face.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <iostream>

FacialAuth::FacialAuth() {
	recognizer = cv::face::LBPHFaceRecognizer::create();
}

FacialAuth::~FacialAuth() {
	// Distruttore: puoi rilasciare risorse se necessario
}

bool FacialAuth::Authenticate(const std::string &user) {
	// Inizializzazione della variabile per l'immagine del volto
	cv::Mat faceImage;

	// Simuliamo il processo di acquisizione del volto (puoi sostituirlo con l'input dalla webcam o immagine)
	faceImage = cv::imread("path_to_user_image.jpg", cv::IMREAD_GRAYSCALE);

	if (faceImage.empty()) {
		std::cerr << "Errore: Immagine del volto non trovata!" << std::endl;
		return false;
	}

	// Carica il modello se non è già stato caricato
	if (!recognizer->isTrained()) {
		if (!LoadModel("path_to_model.xml")) {
			std::cerr << "Errore: Impossibile caricare il modello facciale!" << std::endl;
			return false;
		}
	}

	// Riconosciamo il volto
	return RecognizeFace(faceImage);
}

bool FacialAuth::LoadModel(const std::string &modelPath) {
	// Carica il modello di riconoscimento facciale
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
	// Verifica se il volto è riconosciuto
	int label = -1;
	double confidence = 0.0;

	recognizer->predict(faceImage, label, confidence);

	if (confidence < 50) {  // Soglia di confidenza (puoi cambiarla in base ai tuoi test)
		std::cout << "Autenticazione riuscita! Utente: " << label << std::endl;
		return true;
	} else {
		std::cout << "Autenticazione fallita. Confidenza: " << confidence << std::endl;
		return false;
	}
}
