#include <unistd.h>  // Per geteuid()
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

bool train_model(const std::string& data_dir, const std::string& user_dir) {
	// Creazione del modello
	cv::Ptr<cv::face::LBPHFaceRecognizer> model = cv::face::LBPHFaceRecognizer::create();

	std::vector<cv::Mat> images;
	std::vector<int> labels;

	// Carica le immagini dal directory
	for (const auto& entry : fs::directory_iterator(data_dir)) {
		if (entry.is_directory()) {
			// Carica le immagini in bianco e nero da ciascun subfolder
			cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
			if (img.empty()) {
				std::cerr << "Failed to load image from " << entry.path() << std::endl;
				continue;
			}
			images.push_back(img);
			labels.push_back(std::stoi(entry.path().filename()));  // Usa il nome della directory come label
		}
	}

	if (images.empty()) {
		std::cerr << "No images found for training." << std::endl;
		return false;
	}

	// Train del modello
	model->train(images, labels);

	// Salva il modello
	model->save(user_dir + "/face_model.xml");

	std::cout << "Model trained and saved to " << user_dir << std::endl;
	return true;
}

int main(int argc, char **argv) {
	// Verifica se il programma è eseguito come root (ID utente == 0)
	if (geteuid() != 0) {
		std::cerr << "This program must be run as root!" << std::endl;
		return -1;
	}

	if (argc != 3) {
		std::cerr << "Usage: " << argv[0] << " <data_dir> <user_model_dir>" << std::endl;
		return -1;
	}

	std::string data_dir = argv[1];
	std::string user_dir = argv[2];

	if (!train_model(data_dir, user_dir)) {
		return -1;
	}

	return 0;
}
