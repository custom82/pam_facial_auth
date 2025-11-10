#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;
using namespace cv;
using namespace cv::face;

enum ClassifierType { LBPH, FISHER, EIGEN };

Ptr<FaceRecognizer> loadModel(const std::string& modelPath, ClassifierType classifierType) {
	Ptr<FaceRecognizer> model;

	switch (classifierType) {
		case LBPH:
			model = LBPHFaceRecognizer::create();
			model->read(modelPath);
			break;
		case FISHER:
			model = FisherFaceRecognizer::create();
			model->read(modelPath);
			break;
		case EIGEN:
			model = EigenFaceRecognizer::create();
			model->read(modelPath);
			break;
	}
	return model;
}

// Funzione di test del modello
bool test_model(const std::string& imagePath, const std::string& modelType) {
	Mat testImage = imread(imagePath, IMREAD_GRAYSCALE);
	if (testImage.empty()) {
		std::cerr << "Immagine non trovata!" << std::endl;
		return false;
	}

	ClassifierType classifier = LBPH; // Default

	// Seleziona il classificatore in base all'input
	if (modelType == "fisher") {
		classifier = FISHER;
	} else if (modelType == "eigen") {
		classifier = EIGEN;
	}

	// Carica il modello e fai il riconoscimento
	Ptr<FaceRecognizer> model = loadModel("face_model.xml", classifier);

	int label = -1;
	double confidence = 0.0;
	model->predict(testImage, label, confidence);

	std::cout << "Predizione completata. Utente: " << label << ", Confidenza: " << confidence << std::endl;
	return true;
}

int main(int argc, char** argv) {
	bool noGui = false;
	std::string classifier = "lbph"; // Default classifier

	if (argc < 3) {
		std::cerr << "Uso: " << argv[0] << " <test> <image_path> [--no-gui] [--classifier lbph|fisher|eigen]" << std::endl;
		return 1;
	}

	// Parsing degli argomenti della linea di comando
	for (int i = 1; i < argc; ++i) {
		if (std::string(argv[i]) == "--no-gui") {
			noGui = true;
		} else if (std::string(argv[i]) == "--classifier") {
			if (i + 1 < argc) {
				classifier = argv[++i];
			}
		}
	}

	if (std::string(argv[1]) == "test") {
		if (argc < 4) {
			std::cerr << "Per testare, specifica il percorso dell'immagine!" << std::endl;
			return 1;
		}
		return test_model(argv[2], classifier);
	} else {
		std::cerr << "Comando non valido!" << std::endl;
		return 1;
	}
}
