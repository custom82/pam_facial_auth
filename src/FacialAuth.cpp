#include "FacialAuth.h"
#include "FaceRecWrapper.h"
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <iostream>

int pam_sm_authenticate(pam_handle_t *pamh, int flags, int argc, const char **argv) {
	// Inizializzazione delle variabili per la configurazione
	cv::Mat im;
	int prediction;
	double confidence;
	std::string imagePath = "/path/to/your/image"; // Percorso dell'immagine

	// Carica l'immagine in scala di grigi
	im = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
	if (im.empty()) {
		std::cerr << "Immagine non trovata: " << imagePath << std::endl;
		return PAM_AUTH_ERR;
	}

	// Usa FaceRecWrapper per predire
	FaceRecWrapper faceRecWrapper("/path/to/model", "FaceModel");
	faceRecWrapper.Load("/path/to/training_data");

	// Predizione
	faceRecWrapper.Predict(im, prediction, confidence);

	// Gestisci il risultato della predizione
	if (prediction >= 0) {
		std::cout << "Predizione: " << prediction << " con confidenza: " << confidence << std::endl;
		return PAM_SUCCESS;
	} else {
		std::cerr << "Predizione fallita!" << std::endl;
		return PAM_AUTH_ERR;
	}
}
