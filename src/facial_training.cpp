#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <dirent.h>

namespace fs = std::filesystem;

void train_model(const std::string& method, const std::string& training_data_dir, const std::string& output_file) {
	// Crea il modello
	cv::Ptr<cv::face::FaceRecognizer> model;

	if (method == "lbph") {
		model = cv::face::LBPHFaceRecognizer::create();
	}
	else if (method == "eigen") {
		model = cv::face::EigenFaceRecognizer::create();
	}
	else if (method == "fisher") {
		model = cv::face::FisherFaceRecognizer::create();
	} else {
		std::cerr << "Unknown method: " << method << std::endl;
		return;
	}

	std::vector<cv::Mat> images;
	std::vector<int> labels;

	// Carica le immagini e le etichette
	for (const auto& entry : fs::directory_iterator(training_data_dir)) {
		if (entry.is_directory()) {
			std::string user_dir = entry.path().string();
			for (const auto& img_entry : fs::directory_iterator(user_dir)) {
				if (img_entry.path().extension() == ".jpg" || img_entry.path().extension() == ".png") {
					cv::Mat img = cv::imread(img_entry.path().string(), cv::IMREAD_GRAYSCALE);
					if (img.empty()) {
						continue;
					}
					images.push_back(img);
					labels.push_back(std::stoi(entry.path().filename().string()));  // Usa il nome della cartella come etichetta
				}
			}
		}
	}

	// Verifica se ci sono abbastanza immagini per allenare il modello
	if (images.empty()) {
		std::cerr << "No training images found!" << std::endl;
		return;
	}

	// Allena il modello
	std::cout << "Training with method: " << method << " for user: " << training_data_dir << std::endl;
	std::cout << "Using training data from: " << training_data_dir << std::endl;
	std::cout << "Model will be saved in: " << output_file << std::endl;

	model->train(images, labels);

	// Salva il modello
	model->save(output_file);
	std::cout << "Model saved at: " << output_file << std::endl;
}

int main(int argc, char** argv) {
	if (argc < 5) {
		std::cerr << "Usage: " << argv[0] << " -u <user> -m <method> <training_data_directory> --output <output_directory>" << std::endl;
		return 1;
	}

	std::string user;
	std::string method;
	std::string training_data_dir;
	std::string output_file;

	// Elenco delle opzioni
	for (int i = 1; i < argc; i++) {
		std::string arg = argv[i];
		if (arg == "-u") {
			user = argv[++i];  // Prende l'argomento successivo come nome utente
		} else if (arg == "-m") {
			method = argv[++i];  // Metodo di riconoscimento (lbph, eigen, fisher)
		} else if (arg == "--output" || arg == "-o") {
			output_file = argv[++i];  // Percorso per il salvataggio del modello
		} else {
			training_data_dir = arg;  // Directory contenente i dati di addestramento
		}
	}

	if (user.empty() || method.empty() || training_data_dir.empty() || output_file.empty()) {
		std::cerr << "Missing required arguments!" << std::endl;
		return 1;
	}

	// Aggiungi la directory dell'utente ai dati di addestramento
	training_data_dir += "/" + user;

	// Verifica se la directory dell'utente esiste
	if (!fs::exists(training_data_dir)) {
		std::cerr << "Directory not found for user: " << user << std::endl;
		return 1;
	}

	// Esegui l'addestramento del modello
	train_model(method, training_data_dir, output_file);

	return 0;
}
