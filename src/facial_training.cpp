#include <iostream>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>  // Assicurati di includere OpenCV se necessario

namespace fs = std::filesystem;

bool train_model(const std::string& data_dir, const std::string& user_name) {
	fs::path user_dir = fs::path(data_dir) / user_name;

	// Verifica se la directory dell'utente esiste
	if (!fs::exists(user_dir)) {
		std::cerr << "User directory " << user_dir << " does not exist!" << std::endl;
		return false;
	}

	// Aggiungi la logica per il training del modello
	std::cout << "Training model for user: " << user_name << std::endl;

	// Creazione e salvataggio del modello (assicurati che la logica del modello sia corretta)
	cv::Ptr<cv::face::LBPHFaceRecognizer> model = cv::face::LBPHFaceRecognizer::create();
	model->train(/* immagini, etichette */);
	model->save(user_dir / "face_model.xml");

	return true;
}

int main(int argc, char* argv[]) {
	if (argc != 3) {
		std::cerr << "Usage: " << argv[0] << " <data_dir> <user_name>" << std::endl;
		return 1;
	}

	std::string data_dir = argv[1];
	std::string user_name = argv[2];

	if (!train_model(data_dir, user_name)) {
		return 1;
	}

	std::cout << "Training for user " << user_name << " completed!" << std::endl;
	return 0;
}
