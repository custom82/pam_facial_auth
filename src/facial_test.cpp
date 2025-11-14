#include "../include/libfacialauth.h"
#include <opencv2/face.hpp>
#include <iostream>
#include <unistd.h>

using namespace std;

static void show_help() {
	cout << "Usage: facial_test -u <user> -m <model_path> [options]\n\n"
	<< "Options:\n"
	<< "  -u, --user <user>        Utente da verificare\n"
	<< "  -m, --model <path>       File modello XML\n"
	<< "  -c, --config <file>      File configurazione\n"
	<< "  -d, --device <device>    Webcam (es. /dev/video0)\n"
	<< "  --threshold <value>      Soglia match (default 80.0)\n"
	<< "  --nogui                  Modalità console\n"
	<< "  -v, --verbose            Verbose\n"
	<< "  -h, --help               Mostra help\n";
}

int main(int argc, char *argv[]) {
	string user;
	string model_path;
	string cfgfile = "/etc/pam_facial_auth/pam_facial.conf";
	string device = "/dev/video0";
	bool verbose = false;
	bool nogui = false;
	double threshold = 80.0;

	FacialAuthConfig cfg;

	// ---------------------
	// Parsing argomenti
	// ---------------------
	for (int i = 1; i < argc; i++) {
		string a = argv[i];

		if (a == "-u" || a == "--user")
			user = argv[++i];
		else if (a == "-m" || a == "--model")
			model_path = argv[++i];
		else if (a == "-c" || a == "--config")
			cfgfile = argv[++i];
		else if (a == "-d" || a == "--device")
			device = argv[++i];
		else if (a == "--threshold")
			threshold = stod(argv[++i]);
		else if (a == "--nogui")
			nogui = true;
		else if (a == "-v" || a == "--verbose")
			verbose = true;
		else if (a == "-h" || a == "--help") {
			show_help();
			return 0;
		}
	}

	if (user.empty() || model_path.empty()) {
		cerr << "Parametri mancanti.\n";
		show_help();
		return 1;
	}

	cfg.nogui = nogui;
	cfg.debug = verbose;

	// -----------------------------------------
	// Carica modello
	// -----------------------------------------
	cv::Ptr<cv::face::FaceRecognizer> model =
	cv::face::LBPHFaceRecognizer::create();
	model->read(model_path);

	// -----------------------------------------
	// Carica webcam
	// -----------------------------------------
	cv::VideoCapture cap(0);
	if (!cap.isOpened()) {
		cerr << "Impossibile aprire webcam.\n";
		return 1;
	}

	cv::CascadeClassifier haar;
	cv::dnn::Net dnn;
	bool use_dnn = false;
	string log;

	if (!load_detectors(cfg, haar, dnn, use_dnn, log)) {
		cerr << "Errore: nessun rilevatore disponibile\n";
		return 1;
	}

	if (verbose)
		cout << log;

	cout << "Inizio test...\n";

	while (true) {
		cv::Mat frame;
		cap >> frame;
		if (frame.empty())
			continue;

		cv::Rect face;
		if (!detect_face(cfg, frame, face, haar, dnn))
			continue;

		cv::Mat crop = frame(face);
		cv::Mat gray;
		cv::cvtColor(crop, gray, cv::COLOR_BGR2GRAY);

		int label;
		double conf;
		model->predict(gray, label, conf);

		if (verbose)
			cout << "Confidence: " << conf << endl;

		if (conf < threshold) {
			cout << "VALIDO\n";
			return 0;
		}
	}
}
