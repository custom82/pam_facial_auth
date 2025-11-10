#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <vector>
#include <unistd.h>
#include <pwd.h>

namespace fs = std::filesystem;

void print_help() {
	std::cout << "Uso: facial_training <data_dir> <username>\n";
	std::cout << "<data_dir>  : Percorso alla directory contenente le immagini di addestramento.\n";
	std::cout << "<username>  : Nome dell'utente per il quale abilitare il riconoscimento facciale.\n";
}

bool is_root() {
	return getuid() == 0;  // Verifica se l'utente è root
}

bool train_model(const std::string& data_dir, const std::string& username) {
	std::vector<cv::Mat> images;
	std::vector<int> labels;
	int label = 0;

	// Crea una directory per l'utente se non esiste
	fs::path user_dir = "/var/lib/facial_auth/" + username;
	if (!fs::exists(user_dir)) {
		fs::create_directories(user_dir);
	}

	// Apre la directory
	for (const auto& entry : fs::directory_iterator(data_dir)) {
		if (entry.is_regular_file()) {
			cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
			if (img.empty()) {
				std::cerr << "Errore nel caricare l'immagine: " << entry.path() << std::endl;
				continue;
			}
			images.push_back(img);
			labels.push_back(label);
			label++;

			// Salva l'immagine nell'apposita directory dell'utente
			fs::copy(entry.path(), user_dir / entry.path().filename());
		}
	}

	if (images.empty()) {
		std::cerr << "Nessuna immagine trovata nella directory di addestramento.\n";
		return false;
	}

	// Crea il modello di riconoscimento facciale
	cv::Ptr<cv::face::LBPHFaceRecognizer> model = cv::face::LBPHFaceRecognizer::create();
	model->train(images, labels);

	// Salva il modello
	model->save(user_dir / "face_model.xml");
	std::cout << "Modello di riconoscimento facciale salvato come '" << user_dir / "face_model.xml" << "'.\n";
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

	std::string data_dir = argv[1];
	std::string username = argv[2];

	if (!fs::exists(data_dir) || !fs::is_directory(data_dir)) {
		std::cerr << "La directory specificata non esiste o non è valida.\n";
		return -1;
	}

	return train_model(data_dir, username) ? 0 : -1;
}
