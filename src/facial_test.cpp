#include <iostream>
#include <string>
#include <unistd.h>
#include <getopt.h>

#include "../include/libfacialauth.h"

static void print_usage(const char* prog)
{
	std::cout << "Usage: " << prog << " -u <user> -m <model_path> [options]\n\n"
	<< "Options:\n"
	<< "  -u, --user <user>        Utente da verificare (obbligatorio)\n"
	<< "  -m, --model <path>       File modello XML (obbligatorio)\n"
	<< "  -c, --config <file>      File configurazione (default: /etc/pam_facial_auth/pam_facial.conf)\n"
	<< "  -d, --device <device>    Dispositivo webcam (es. /dev/video0)\n"
	<< "      --threshold <value>  Soglia confidenza match (default: 80.0)\n"
	<< "  -v, --verbose            Output verboso\n"
	<< "      --nogui              Disabilita GUI\n"
	<< "  -h, --help               Mostra questo messaggio\n";
}

int main(int argc, char** argv)
{
	std::string user;
	std::string model_path;
	std::string config_path = "/etc/pam_facial_auth/pam_facial.conf";

	// Config di default caricata via libfacialauth
	FacialAuthConfig cfg;

	// Tabelle opzioni lunghe
	static struct option long_opts[] = {
		{"user",      required_argument, 0, 'u'},
		{"model",     required_argument, 0, 'm'},
		{"config",    required_argument, 0, 'c'},
		{"device",    required_argument, 0, 'd'},
		{"threshold", required_argument, 0, 1000},
		{"verbose",   no_argument,       0, 'v'},
		{"nogui",     no_argument,       0, 1001},
		{"help",      no_argument,       0, 'h'},
		{0,0,0,0}
	};

	int opt, longidx;
	while ((opt = getopt_long(argc, argv, "u:m:c:d:vh", long_opts, &longidx)) != -1)
	{
		switch (opt)
		{
			case 'u':
				user = optarg;
				break;

			case 'm':
				model_path = optarg;
				break;

			case 'c':
				config_path = optarg;
				break;

			case 'd':
				cfg.device = optarg;
				break;

			case 1000: // --threshold
				cfg.threshold = atof(optarg);
				break;

			case 'v':
				cfg.debug = true;
				break;

			case 1001: // --nogui
				cfg.nogui = true;
				break;

			case 'h':
				print_usage(argv[0]);
				return 0;

			default:
				print_usage(argv[0]);
				return 1;
		}
	}

	// Check obbligatori
	if (user.empty() || model_path.empty()) {
		std::cerr << "Errore: -u <user> e -m <model_path> sono obbligatori.\n";
		print_usage(argv[0]);
		return 1;
	}

	// Carica configurazione globale
	std::string log;
	read_kv_config(config_path, cfg, &log);
	if (cfg.debug) std::cerr << log;

	// -------------------------
	// Inizializza wrapper volto
	// -------------------------
	FaceRecWrapper faceRec(cfg.model_path, user, cfg.model);

	if (!file_exists(model_path)) {
		std::cerr << "Errore: modello non trovato: " << model_path << "\n";
		return 1;
	}

	faceRec.Load(model_path);

	// -------------------------
	// Apri la webcam
	// -------------------------
	cv::VideoCapture cap;
	std::string used_device;

	if (!open_camera(cfg, cap, used_device)) {
		std::cerr << "Errore: impossibile aprire la webcam (" << cfg.device << ")\n";
		return 1;
	}

	if (cfg.debug)
		std::cerr << "[DEBUG] Webcam aperta: " << used_device << "\n";

	cap.set(cv::CAP_PROP_FRAME_WIDTH, cfg.width);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.height);

	// -------------------------
	// RILEVAMENTO: usa la funzione utility detect_face()
	// -------------------------
	cv::CascadeClassifier haar;
	cv::dnn::Net dnn;
	bool use_dnn = false;
	load_detectors(cfg, haar, dnn, use_dnn, log);

	if (cfg.debug) std::cerr << log;

	cv::Mat frame;
	cv::Rect face_roi;

	std::cout << "Premi 'q' per uscire.\n";

	while (true) {
		cap >> frame;
		if (frame.empty()) continue;

		// Detect volto
		if (detect_face(cfg, frame, face_roi, haar, dnn))
		{
			cv::Mat face = frame(face_roi).clone();

			int predicted = -1;
			double conf = 9999.0;
			faceRec.Predict(face, predicted, conf);

			std::cout << "User=" << predicted
			<< " Conf=" << conf
			<< " (threshold=" << cfg.threshold << ")\n";

			if (conf < cfg.threshold)
				std::cout << "MATCH ✔️\n";
			else
				std::cout << "NO MATCH ❌\n";

			if (!cfg.nogui)
				cv::rectangle(frame, face_roi, cv::Scalar(0,255,0), 2);
		}

		if (!cfg.nogui) {
			cv::imshow("facial_test", frame);
			if ((char)cv::waitKey(1) == 'q') break;
		}
	}

	return 0;
}
