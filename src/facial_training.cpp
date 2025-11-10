#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

// Funzione per addestrare il modello
void train_model(const std::string& user, const std::string& method, const std::string& training_data_directory, const std::string& output_model_path) {
	// Verifica che la directory di dati dell'utente esista
	std::string user_data_dir = training_data_directory + "/" + user;
	if (!fs::exists(user_data_dir)) {
		std::cerr << "Directory not found for user: " << user << std::endl;
		return;
	}

	// Aggiungi la logica per supportare i vari metodi
	std::cout << "Training with method: " << method << " for user: " << user << std::endl;
	std::cout << "Using training data from: " << user_data_dir << std::endl;
	std::cout << "Model will be saved in: " << output_model_path << std::endl;

	// Carica le immagini per l'addestramento
	std::vector<cv::Mat> images;
	std::vector<int> labels;
	int label = 0;  // Usato per etichettare tutte le immagini dello stesso utente

	// Cicla attraverso le immagini nella directory dell'utente
	for (const auto &entry : fs::directory_iterator(user_data_dir)) {
		if (entry.is_regular_file() && (entry.path().extension() == ".jpg" || entry.path().extension() == ".png")) {
			// Carica l'immagine come immagine in scala di grigi
			cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
			if (img.empty()) {
				std::cerr << "Unable to read image: " << entry.path() << std::endl;
			} else {
				images.push_back(img);
				labels.push_back(label);  // Assegna l'etichetta corretta
			}
		}
	}

	// Verifica che ci siano immagini per l'addestramento
	if (images.empty()) {
		std::cerr << "No images loaded for training!" << std::endl;
		return;
	}

	// Inizializza il modello in base al metodo scelto
	cv::Ptr<cv::face::FaceRecognizer> model;
	if (method == "lbph") {
		model = cv::face::LBPHFaceRecognizer::create();
	} else if (method == "eigen") {
		model = cv::face::EigenFaceRecognizer::create();
	} else if (method == "fisher") {
		model = cv::face::FisherFaceRecognizer::create();
	} else {
		std::cerr << "Invalid method specified! Choose from lbph, eigen, or fisher." << std::endl;
		return;
	}

	// Addestra il modello con le immagini caricate
	model->train(images, labels);

	// Salva il modello addestrato
	model->save(output_model_path);
	std::cout << "Model saved at: " << output_model_path << std::endl;
}

int main(int argc, char* argv[]) {
	// Verifica se sono stati forniti i parametri
	if (argc < 5) {
		std::cerr << "Usage: facial_training -u <user> -m <method> <training_data_directory> --output <output_model_path>" << std::endl;
		return -1;
	}

	std::string user;
	std::string method;
	std::string training_data_directory;
	std::string output_model_path;

	// Parsing dei parametri
	for (int i = 1; i < argc; i++) {
		if (std::string(argv[i]) == "-u") {
			user = argv[++i];
		} else if (std::string(argv[i]) == "-m") {
			method = argv[++i];
		} else if (std::string(argv[i]) == "--output") {
			output_model_path = argv[++i];
		} else {
			training_data_directory = argv[i];
		}
	}

	// Esegui il training del modello
	train_model(user, method, training_data_directory, output_model_path);

	return 0;
}
