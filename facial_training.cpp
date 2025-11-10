#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>
#include <filesystem>
#include <string>

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

// Funzione di addestramento
bool train_model(const std::string& dataDir, const std::string& modelType) {
	std::vector<Mat> images;
	std::vector<int> labels;

	// Leggi le immagini da dataDir (assumiamo che i nomi siano nel formato: "nomeUtente_XX.jpg")
	for (const auto& entry : fs::directory_iterator(dataDir)) {
		if (entry.path().extension() == ".jpg") {
			Mat img = imread(entry.path().string(), IMREAD_GRAYSCALE);
			if (!img.empty()) {
				images.push_back(img);
				labels.push_back(std::stoi(entry.path().stem().string())); // Prendi l'ID utente dal nome del file
			}
		}
	}

	ClassifierType classifier = LBPH; // Default

	// Seleziona il classificatore in base all'input
	if (modelType == "fisher") {
		classifier = FISHER;
	} else if (modelType == "eigen") {
		classifier = EIGEN;
	}

	// Allenamento del modello
	Ptr<FaceRecognizer> model = loadModel("face_model.xml", classifier);
	model->train(images, labels);
	model->save("face_model.xml");

	std::cout << "Modello allenato e salvato con successo!" << std::endl;
	return true;
}

int main(int argc, char** argv) {
	bool noGui = false;
	std::string classifier = "lbph"; // Default classifier

	if (argc < 3) {
		std::cerr << "Uso: " << argv[0] << " <train> <data_dir> [--no-gui] [--classifier lbph|fisher|eigen]" << std::endl;
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

	if (std::string(argv[1]) == "train") {
		if (argc < 4) {
			std::cerr << "Per allenare, specifica la directory dei dati!" << std::endl;
			return 1;
		}
		return train_model(argv[2], classifier);
	} else {
		std::cerr << "Comando non valido!" << std::endl;
		return 1;
	}
}
