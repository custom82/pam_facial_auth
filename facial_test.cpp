#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>
#include <unistd.h>
#include <filesystem>

namespace fs = std::filesystem;

void print_help() {
	std::cout << "Uso: facial_test <image_path> <username>\n";
	std::cout << "<image_path> : Percorso all'immagine da testare con il modello.\n";
	std::cout << "<username> : Nome dell'utente per il quale eseguire il riconoscimento facciale.\n";
}

bool is_root() {
	return getuid() == 0;  // Verifica se l'utente è root
}

bool test_model(const std::string& image_path, const std::string& username) {
	// Carica il modello pre-addestrato
	cv::Ptr<cv::face::LBPHFaceRecognizer> model = cv::face::LBPHFaceRecognizer::create();
	std::string model_path = "/var/lib/facial_auth/" + username + "/face_model.xml";

	// Verifica che il modello esista
	if (!fs::exists(model_path)) {
		std::cerr << "Il modello per l'utente " << username << " non esiste.\n";
		return false;
	}

	model->read(model_path);

	// Carica l'immagine di test
	cv::Mat test_image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
	if (test_image.empty()) {
		std::cerr << "Immagine non trovata o non valida: " << image_path << std::endl;
		return false;
	}

	// Riconoscimento facciale
	int predicted_label;
	double confidence;
	model->predict(test_image, predicted_label, confidence);

	std::cout << "Risultato del test per l'utente " << username << ": etichetta prevista = "
	<< predicted_label << ", confidenza = " << confidence << "\n";

	return true;
}

int main(int argc, char** argv) {
	if (argc != 3) {
		print_help();
		return -1;
	}

	if (!is_root()) {
		std::cerr << "Questo programma deve essere eseguito come root.\n";
		return -1;
	}

	std::string image_path = argv[1];
	std::string username = argv[2];

	return test_model(image_path, username) ? 0 : -1;
}
