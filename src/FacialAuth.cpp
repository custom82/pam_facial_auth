#include "FacialAuth.h"
#include "FaceRecWrapper.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

int pam_sm_authenticate(pam_handle_t *pamh, int flags, int argc, const char **argv) {
	// Inizializza il riconoscimento facciale
	std::string modelPath = "/path/to/model.yml";
	std::string name = "Face Recognizer";

	// Crea oggetto FaceRecWrapper
	FaceRecWrapper frw(modelPath, name);

	// Carica il modello
	frw.Load(modelPath);

	// Carica l'immagine da testare
	cv::Mat image = cv::imread("/path/to/image.jpg");
	if (image.empty()) {
		std::cerr << "Immagine non trovata!" << std::endl;
		return PAM_AUTH_ERR;
	}

	// Riconoscimento facciale
	int prediction = -1;
	double confidence = 0.0;
	frw.Predict(image, prediction, confidence);

	std::cout << "Predizione: " << prediction << ", Confidenza: " << confidence << std::endl;

	return PAM_SUCCESS; // O PAM_AUTH_ERR in caso di errore
}
