#include "../include/libfacialauth.h"
#include <opencv2/face.hpp>
#include <iostream>
#include <filesystem>

using namespace std;
namespace fs = std::filesystem;

static void show_help() {
	cout << "Usage: facial_training -u <user> -m <method> <training_dir> [options]\n\n"
	<< "Options:\n"
	<< "  -u, --user <name>        Username da addestrare\n"
	<< "  -m, --method <type>      Metodo (lbph, eigen, fisher)\n"
	<< "  -o, --output <file>      Percorso file modello XML\n"
	<< "  -f, --force              Sovrascrive file modello esistente\n"
	<< "  -v, --verbose            Output dettagliato\n"
	<< "  -h, --help               Mostra help\n\n"
	<< "Esempi:\n"
	<< "  facial_training -u custom -m lbph /etc/pam_facial_auth/\n"
	<< "  facial_training -u custom -m lbph /etc/pam_facial_auth/ -o /etc/pam_facial_auth/custom.xml\n";
}

int main(int argc, char *argv[]) {
	string user;
	string method;
	string training_dir;
	string output_file;
	bool verbose = false;
	bool force = false;

	// ---------------------
	// Parsing argomenti
	// ---------------------
	for (int i = 1; i < argc; i++) {
		string a = argv[i];

		if (a == "-u" || a == "--user")
			user = argv[++i];
		else if (a == "-m" || a == "--method")
			method = argv[++i];
		else if (a == "-o" || a == "--output")
			output_file = argv[++i];
		else if (a == "-f" || a == "--force")
			force = true;
		else if (a == "-v" || a == "--verbose")
			verbose = true;
		else if (a == "-h" || a == "--help") {
			show_help();
			return 0;
		}
		else
			training_dir = a;
	}

	if (user.empty() || method.empty() || training_dir.empty()) {
		cerr << "Errore: parametri insufficienti.\n";
		show_help();
		return 1;
	}

	if (output_file.empty())
		output_file = "/etc/pam_facial_auth/" + user + "/models/model.xml";

	if (fs::exists(output_file) && !force) {
		cerr << "File modello esistente. Usa --force.\n";
		return 1;
	}

	// ---------------------
	// Carica immagini
	// ---------------------
	vector<cv::Mat> images;
	vector<int> labels;

	string img_dir = training_dir + "/" + user + "/images";
	for (auto &p : fs::directory_iterator(img_dir)) {
		if (p.path().extension() == ".png") {
			images.push_back(cv::imread(p.path().string(), cv::IMREAD_GRAYSCALE));
			labels.push_back(1);
			if (verbose)
				cout << "Caricata: " << p.path() << endl;
		}
	}

	if (images.empty()) {
		cerr << "Nessuna immagine trovata in: " << img_dir << "\n";
		return 1;
	}

	// ---------------------
	// Seleziona metodo
	// ---------------------
	cv::Ptr<cv::face::FaceRecognizer> model;

	if (method == "lbph")
		model = cv::face::LBPHFaceRecognizer::create();
	else if (method == "eigen")
		model = cv::face::EigenFaceRecognizer::create();
	else if (method == "fisher")
		model = cv::face::FisherFaceRecognizer::create();
	else {
		cerr << "Metodo sconosciuto.\n";
		return 1;
	}

	model->train(images, labels);
	fs::create_directories(fs::path(output_file).parent_path());
	model->save(output_file);

	cout << "Modello salvato in: " << output_file << endl;

	return 0;
}
