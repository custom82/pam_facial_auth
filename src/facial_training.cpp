#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>  // Include il modulo face
#include <iostream>
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

// Funzione per addestrare il modello
bool train_model(const std::string& data_dir, const std::string& user_dir) {
	// Carica le immagini da addestrare
	std::vector<cv::Mat> images;
	std::vector<int> labels;

	// Funzione per caricare le immagini da un dato percorso
	for (const auto& entry : fs::directory_iterator(data_dir)) {
		// Assicurati che il file sia un'immagine
		if (entry.is_regular_file() && entry.path().extension() == ".jpg") {
			cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);  // Legge in scala di grigi
			if (!img.empty()) {
				images.push_back(img);
				labels.push_back(std::stoi(entry.path().stem().string()));  // Supponiamo che il nome del file contenga l'etichetta
			}
		}
	}

	if (images.empty()) {
		std::cerr << "No images found for training!" << std::endl;
		return false;
	}

	// Aggiungi il modello LBPH
	cv::Ptr<cv::face::LBPHFaceRecognizer> model = cv::face::LBPHFaceRecognizer::create();

	// Addestra il modello
	model->train(images, labels);

	// Salva il modello addestrato nella directory dell'utente
	model->save(user_dir + "/face_model.xml");

	std::cout << "Model trained and saved to " << user_dir + "/face_model.xml" << std::endl;
	return true;
}

int main(int argc, char** argv) {
	if (argc != 3) {
		std::cerr << "Usage: " << argv[0] << " <data_directory> <user_model_directory>" << std::endl;
		return -1;
	}

	std::string data_dir = argv[1];
	std::string user_dir = argv[2];

	if (train_model(data_dir, user_dir)) {
		std::cout << "Training successful!" << std::endl;
		return 0;
	} else {
		std::cerr << "Training failed!" << std::endl;
		return -1;
	}
}
