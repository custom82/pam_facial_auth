#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>  // Modulo facciale
#include <iostream>
#include <filesystem>
#include <getopt.h>  // Libreria per il parsing degli argomenti da linea di comando

namespace fs = std::filesystem;

void print_usage() {
	std::cout << "Usage: facial_training -u <user> -m <method> <training_data_directory> [--output <output_directory>]" << std::endl;
	std::cout << "Methods: lbph, eigen, fisher" << std::endl;
}

bool train_model(const std::string& user, const std::string& training_data_dir, const std::string& output_dir, const std::string& method) {
	// Inizializza la lista delle immagini e le etichette
	std::vector<cv::Mat> images;
	std::vector<int> labels;

	fs::path user_dir = fs::path(training_data_dir) / user;
	if (!fs::exists(user_dir)) {
		std::cerr << "User directory does not exist: " << user_dir << std::endl;
		return false;
	}

	// Leggi tutte le immagini della directory dell'utente
	for (const auto& entry : fs::directory_iterator(user_dir)) {
		if (entry.is_regular_file() && (entry.path().extension() == ".jpg" || entry.path().extension() == ".png")) {
			cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
			if (!img.empty()) {
				images.push_back(img);
				labels.push_back(1);  // Usa un'etichetta fissa, oppure un'altra logica per differenti etichette
			}
		}
	}

	// Verifica se ci sono abbastanza immagini
	if (images.size() < 2) {
		std::cerr << "Insufficient images for training (at least 2 required)" << std::endl;
		return false;
	}

	// Crea e allena il modello in base al metodo scelto
	cv::Ptr<cv::face::FaceRecognizer> model;

	if (method == "lbph") {
		model = cv::face::LBPHFaceRecognizer::create();
	} else if (method == "eigen") {
		model = cv::face::EigenFaceRecognizer::create();
	} else if (method == "fisher") {
		model = cv::face::FisherFaceRecognizer::create();
	} else {
		std::cerr << "Unknown method: " << method << std::endl;
		return false;
	}

	model->train(images, labels);

	// Salva il modello nel percorso di output
	fs::path model_path = fs::path(output_dir) / (user + "_face_model.xml");
	model->save(model_path.string());

	std::cout << "Model trained and saved at: " << model_path << std::endl;
	return true;
}

int main(int argc, char** argv) {
	// Variabili per il parsing degli argomenti
	std::string user;
	std::string output_dir = "/etc/pam_facial_auth";  // Default output directory
	std::string method = "lbph";  // Metodo di riconoscimento facciale di default

	int opt;
	while ((opt = getopt(argc, argv, "u:m:")) != -1) {
		switch (opt) {
			case 'u':
				user = optarg;  // Assegna l'utente specificato con il parametro -u
				break;
			case 'm':
				method = optarg;  // Assegna il metodo specificato con il parametro -m
				break;
			default:
				print_usage();
				return 1;
		}
	}

	// Verifica se l'argomento dell'utente è stato passato
	if (user.empty() || optind >= argc) {
		std::cerr << "User (-u) and training directory are required!" << std::endl;
		print_usage();
		return 1;
	}

	std::string training_data_dir = argv[optind];  // Directory di dati di addestramento

	// Gestisce il parametro di output
	if (optind + 1 < argc && std::string(argv[optind + 1]).substr(0, 2) == "--") {
		output_dir = argv[optind + 1];
	}

	// Avvia il processo di addestramento
	bool success = train_model(user, training_data_dir, output_dir, method);
	return success ? 0 : 1;
}
