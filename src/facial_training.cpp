#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>
#include <filesystem>
#include <getopt.h>  // Per la gestione degli argomenti della riga di comando

namespace fs = std::filesystem;

void print_usage() {
	std::cout << "Uso: facial_training [opzioni] <percorso_immagini> <percorso_salvataggio>\n"
	<< "Opzioni:\n"
	<< "  -o, --output <percorso>  Specifica dove salvare il modello\n"
	<< "  -h, --help               Mostra questo aiuto\n";
}

bool train_model(const std::string& images_path, const std::string& model_save_path) {
	// Carica le immagini e le etichette per l'addestramento
	std::vector<cv::Mat> images;
	std::vector<int> labels;

	// Qui dovrebbe esserci la logica per caricare le immagini e le etichette (non implementato in questo esempio)
	// Esempio: carica le immagini in `images` e le etichette in `labels`

	// Creazione del riconoscitore facciale LBPH
	cv::Ptr<cv::face::LBPHFaceRecognizer> model = cv::face::LBPHFaceRecognizer::create();

	// Addestra il modello
	model->train(images, labels);

	// Salva il modello
	model->save(model_save_path);
	return true;
}

int main(int argc, char *argv[]) {
	std::string images_path;
	std::string model_save_path = "/default/path/to/model.xml";  // Percorso di default per il salvataggio

	// Parsing degli argomenti della riga di comando
	static struct option long_options[] = {
		{"output", required_argument, 0, 'o'},
		{"help", no_argument, 0, 'h'},
		{0, 0, 0, 0}
	};

	int option_index = 0;
	int c;

	while ((c = getopt_long(argc, argv, "o:h", long_options, &option_index)) != -1) {
		switch (c) {
			case 'o':
				model_save_path = optarg;
				break;
			case 'h':
				print_usage();
				return 0;
			default:
				print_usage();
				return 1;
		}
	}

	if (optind + 2 > argc) {
		std::cerr << "Errore: è necessario fornire sia il percorso delle immagini che il percorso di salvataggio.\n";
		print_usage();
		return 1;
	}

	images_path = argv[optind];  // Il primo parametro è il percorso delle immagini
	model_save_path = argv[optind + 1];  // Il secondo parametro è il percorso di salvataggio del modello

	if (!fs::exists(images_path)) {
		std::cerr << "Errore: il percorso delle immagini non esiste: " << images_path << std::endl;
		return 1;
	}

	std::cout << "Addestramento in corso. Il modello sarà salvato in: " << model_save_path << std::endl;

	if (!train_model(images_path, model_save_path)) {
		std::cerr << "Errore nell'addestramento del modello.\n";
		return 1;
	}

	std::cout << "Modello addestrato e salvato con successo in: " << model_save_path << std::endl;
	return 0;
}
