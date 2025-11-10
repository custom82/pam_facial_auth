#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <string>
#include <vector>
#include <getopt.h>

namespace fs = std::filesystem;
using namespace cv;

void print_help() {
	std::cout << "Usage: facial_training -u <user> -m <method> <training_data_directory> [--output <output_directory>]\n";
	std::cout << "Methods: lbph, eigen, fisher\n";
}

int main(int argc, char *argv[]) {
	// Variabili per i parametri
	std::string user;
	std::string method = "lbph"; // Default method
	std::string training_data_dir;
	std::string output_model_file;

	// Parsing dei parametri
	static struct option long_options[] = {
		{"user", required_argument, 0, 'u'},
		{"method", required_argument, 0, 'm'},
		{"output", required_argument, 0, 'o'},
		{0, 0, 0, 0}
	};

	int option_index = 0;
	int c;
	while ((c = getopt_long(argc, argv, "u:m:o:", long_options, &option_index)) != -1) {
		switch (c) {
			case 'u':
				user = optarg;
				break;
			case 'm':
				method = optarg;
				break;
			case 'o':
				output_model_file = optarg;
				break;
			default:
				print_help();
				return 1;
		}
	}

	if (optind >= argc) {
		std::cerr << "Missing training data directory\n";
		print_help();
		return 1;
	}

	training_data_dir = argv[optind];

	if (user.empty() || training_data_dir.empty()) {
		std::cerr << "Error: Missing required arguments\n";
		print_help();
		return 1;
	}

	std::cout << "Training with method: " << method << " for user: " << user << "\n";
	std::cout << "Using training data from: " << training_data_dir << "\n";
	std::cout << "Model will be saved in: " << output_model_file << "\n";

	// Prepara il classificatore
	Ptr<face::FaceRecognizer> model;

	if (method == "lbph") {
		model = face::LBPHFaceRecognizer::create();
	} else if (method == "eigen") {
		model = face::EigenFaceRecognizer::create();
	} else if (method == "fisher") {
		model = face::FisherFaceRecognizer::create();
	} else {
		std::cerr << "Unsupported method: " << method << "\n";
		return 1;
	}

	// Carica le immagini e le etichette di addestramento
	std::vector<Mat> images;
	std::vector<int> labels;
	// Esegui la lettura delle immagini e l'assegnazione delle etichette (come hai già fatto nel codice precedente)
	// Aggiungi la logica di caricamento delle immagini e salvataggio del modello

	// Salva il modello nella directory specificata
	if (!output_model_file.empty()) {
		model->save(output_model_file);
		std::cout << "Model saved at: " << output_model_file << std::endl;
	}

	return 0;
}
