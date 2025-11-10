#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <filesystem>
#include <string>
#include <vector>
#include <dirent.h>

namespace fs = std::filesystem;

void train_model(const std::string& user, const std::string& method, const std::string& training_data_dir, const std::string& output_file) {
	std::cout << "Training with method: " << method << " for user: " << user << std::endl;
	std::cout << "Using training data from: " << training_data_dir << std::endl;
	std::cout << "Model will be saved in: " << output_file << std::endl;

	// Percorso completo della directory dell'utente
	std::string user_dir = training_data_dir + "/" + user;
	std::cout << "Verificando la directory: " << user_dir << std::endl; // Log per il debug

	// Verifica che la directory esista
	if (!fs::exists(user_dir)) {
		std::cerr << "Directory not found for user: " << user << std::endl;
		return;
	}

	std::vector<cv::Mat> images;
	std::vector<int> labels;

	// Leggi tutte le immagini nella directory dell'utente
	for (const auto& entry : fs::directory_iterator(user_dir)) {
		if (entry.is_regular_file() && (entry.path().extension() == ".jpg" || entry.path().extension() == ".png")) {
			cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);  // Carica immagine in scala di grigi
			if (img.empty()) {
				std::cerr << "Errore nel caricare l'immagine: " << entry.path() << std::endl;
				continue;
			}
			images.push_back(img);

			// Aggiungi l'etichetta
			labels.push_back(0);  // Qui si potrebbe usare l'ID dell'utente o un altro valore
		}
	}

	if (images.empty()) {
		std::cerr << "No valid images found in the directory." << std::endl;
		return;
	}

	cv::Ptr<cv::face::LBPHFaceRecognizer> model;
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

	// Allena il modello
	model->train(images, labels);

	// Salva il modello
	model->save(output_file);
	std::cout << "Model saved at: " << output_file << std::endl;
}

int main(int argc, char** argv) {
	if (argc < 5) {
		std::cerr << "Usage: " << argv[0] << " -u <user> -m <method> <training_data_directory> --output <output_file>" << std::endl;
		return 1;
	}

	std::string user;
	std::string method;
	std::string training_data_dir;
	std::string output_file;

	// Parsing dei parametri
	for (int i = 1; i < argc; i++) {
		std::string arg = argv[i];
		if (arg == "-u" || arg == "--user") {
			user = argv[++i];
		} else if (arg == "-m" || arg == "--method") {
			method = argv[++i];
		} else if (arg == "--output") {
			output_file = argv[++i];
		} else {
			training_data_dir = arg;  // Il primo parametro rimanente è la directory dei dati di addestramento
		}
	}

	if (user.empty() || method.empty() || training_data_dir.empty() || output_file.empty()) {
		std::cerr << "Missing required arguments!" << std::endl;
		return 1;
	}

	train_model(user, method, training_data_dir, output_file);
	return 0;
}
