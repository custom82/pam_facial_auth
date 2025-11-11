#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <unistd.h>  // per access()

using namespace cv;
using namespace std;

// === CONFIG ===
static const string DEFAULT_CONFIG_PATH = "/etc/pam_facial_auth/pam_facial.conf";

// === STRUTTURA CONFIG ===
struct Config {
	string device;
	int width = 640;
	int height = 480;
};

// === LETTURA CONFIGURAZIONE ===
Config load_config(const string& config_path) {
	Config cfg;
	ifstream file(config_path);
	if (!file.is_open()) {
		cerr << "[WARN] Impossibile aprire il file di configurazione: " << config_path << endl;
		return cfg;
	}

	string line;
	while (getline(file, line)) {
		if (line.empty() || line[0] == '#') continue;
		string key, value;
		stringstream ss(line);
		if (getline(ss, key, '=') && getline(ss, value)) {
			if (key == "device") cfg.device = value;
			else if (key == "width") cfg.width = stoi(value);
			else if (key == "height") cfg.height = stoi(value);
		}
	}

	return cfg;
}

// === OPZIONI CLI ===
struct Options {
	string user;
	string model_path;
	string config_path = DEFAULT_CONFIG_PATH;
	string device_override;
	double threshold = 80.0;  // Soglia predefinita
	bool verbose = false;
	bool nogui = false;  // Modalità senza GUI
};

// === HELP ===
void show_help() {
	cout << "Usage: facial_test -u <user> -m <model_path> [options]\n\n"
	<< "Options:\n"
	<< "  -u, --user <user>        Utente da verificare (obbligatorio)\n"
	<< "  -m, --model <path>       File modello XML (obbligatorio)\n"
	<< "  -c, --config <file>      File di configurazione (default: /etc/pam_facial_auth/pam_facial.conf)\n"
	<< "  -d, --device <device>    Dispositivo webcam (es. /dev/video0)\n"
	<< "  --threshold <value>      Soglia di confidenza per il match (default: 80.0)\n"
	<< "  -v, --verbose            Modalità verbosa\n"
	<< "  --nogui                  Disabilita la GUI (solo console)\n"
	<< "  -h, --help               Mostra questo messaggio\n"
	<< endl;
}

// === PARSING ARGOMENTI ===
Options parse_args(int argc, char** argv) {
	Options opt;
	for (int i = 1; i < argc; ++i) {
		string arg = argv[i];
		if (arg == "-u" || arg == "--user") {
			if (++i < argc) opt.user = argv[i];
		} else if (arg == "-m" || arg == "--model") {
			if (++i < argc) opt.model_path = argv[i];
		} else if (arg == "-c" || arg == "--config") {
			if (++i < argc) opt.config_path = argv[i];
		} else if (arg == "-d" || arg == "--device") {
			if (++i < argc) opt.device_override = argv[i];
		} else if (arg == "--threshold") {
			if (++i < argc) opt.threshold = stod(argv[i]);
		} else if (arg == "-v" || arg == "--verbose") {
			opt.verbose = true;
		} else if (arg == "--nogui") {
			opt.nogui = true;
		} else if (arg == "-h" || arg == "--help") {
			show_help();
			exit(0);
		}
	}
	return opt;
}

// === MAIN ===
int main(int argc, char** argv) {
	Options opt = parse_args(argc, argv);

	if (opt.user.empty() || opt.model_path.empty()) {
		cerr << "[ERROR] Parametri obbligatori mancanti (-u e -m).\n";
		show_help();
		return 1;
	}

	if (opt.verbose) {
		cout << "[INFO] Utente: " << opt.user << endl;
		cout << "[INFO] Modello: " << opt.model_path << endl;
		cout << "[INFO] Config:  " << opt.config_path << endl;
	}

	// Carica config
	Config cfg = load_config(opt.config_path);
	string webcam_device = opt.device_override.empty() ? cfg.device : opt.device_override;
	if (webcam_device.empty()) webcam_device = "/dev/video0";

	if (opt.verbose)
		cout << "[INFO] Webcam device: " << webcam_device
		<< " (" << cfg.width << "x" << cfg.height << ")\n";

	// Verifica accesso al device
	if (access(webcam_device.c_str(), F_OK) != 0) {
		cerr << "[ERROR] Il dispositivo " << webcam_device << " non esiste o non è accessibile.\n";
		return 1;
	}

	// Carica modello LBPH
	Ptr<face::LBPHFaceRecognizer> model = face::LBPHFaceRecognizer::create();
	try {
		model->read(opt.model_path);
	} catch (const cv::Exception& e) {
		cerr << "[ERROR] Impossibile caricare il modello: " << e.what() << endl;
		return 1;
	}

	// Apertura webcam
	VideoCapture cap(webcam_device, cv::CAP_V4L2);
	if (!cap.isOpened()) {
		cerr << "[ERROR] Impossibile aprire il dispositivo " << webcam_device << endl;
		return 1;
	}
	cap.set(CAP_PROP_FRAME_WIDTH, cfg.width);
	cap.set(CAP_PROP_FRAME_HEIGHT, cfg.height);

	CascadeClassifier face_cascade("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml");
	if (face_cascade.empty()) {
		cerr << "[ERROR] Impossibile caricare il classificatore Haar per il rilevamento dei volti.\n";
		return 1;
	}

	if (opt.verbose)
		cout << "[INFO] Webcam pronta, in attesa del volto...\n";

	Mat frame, gray;
	while (true) {
		cap >> frame;
		if (frame.empty()) break;
		cvtColor(frame, gray, COLOR_BGR2GRAY);

		vector<Rect> faces;
		face_cascade.detectMultiScale(gray, faces, 1.1, 5, 0, Size(100, 100));

		for (const auto& face : faces) {
			Mat faceROI = gray(face);
			int label = -1;
			double confidence = 0.0;
			model->predict(faceROI, label, confidence);

			bool recognized = (confidence < opt.threshold);
			Scalar color = recognized ? Scalar(0,255,0) : Scalar(0,0,255);
			rectangle(frame, face, color, 2);

			string text = recognized ? "Match: " + opt.user : "Unknown";
			putText(frame, text, Point(face.x, face.y - 10), FONT_HERSHEY_SIMPLEX, 0.8, color, 2);

			if (opt.verbose)
				cout << "[DEBUG] Confidence: " << confidence << " → "
				<< (recognized ? "MATCH" : "NO MATCH") << endl;

			if (recognized) {
				cout << "[SUCCESS] Utente " << opt.user << " riconosciuto con confidenza " << confidence << endl;
				return 0;
			}
		}

		if (!opt.nogui) {
			imshow("Facial Verification", frame);
			if (waitKey(1) == 27) break; // ESC per uscire
		}
	}

	cout << "[FAILURE] Nessuna corrispondenza trovata per l'utente " << opt.user << endl;
	return 1;
}
