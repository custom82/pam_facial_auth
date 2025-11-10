PAM_EXTERN int pam_sm_authenticate(pam_handle_t *pamh, int flags, int argc, const char **argv) {
	// Prima verifica se possiamo ottenere l'username
	const char * user;
	int ret = pam_get_user(pamh, &user, "Username: ");
	if (ret != PAM_SUCCESS) {
		return ret;
	}
	std::string username(user);

	// Carica la configurazione
	std::map<std::string, std::string> config;
	Utils::GetConfig("/etc/pam-facial-auth/config", config);

	// Imposta i parametri dal file di configurazione
	std::time_t timeout = std::stoi(config["timeout"]);
	double threshold = std::stod(config["threshold"]);
	bool imCapture = config["imageCapture"] == "true";

	// Crea un oggetto FaceRecWrapper passando il percorso del modello e il nome della tecnica
	FaceRecWrapper frw("/etc/pam-facial-auth/model", "eigen");

	// Setup / controllo del ciclo
	std::time_t start = std::time(nullptr);
	std::string imagePathLast;
	cv::VideoCapture vc;
	if (imCapture && !vc.open(0)) {
		std::cout << "Could not open camera." << std::endl;
		return PAM_AUTH_ERR;
	}
	std::cout << "Starting facial recognition for " << username << "..." << std::endl;

	while (std::time(nullptr) - start < timeout) {
		cv::Mat im;

		if (imCapture) {  // Cattura l'immagine attivamente
			vc.read(im);
			cv::cvtColor(im, im, cv::COLOR_BGR2GRAY);
		} else {  // Controlla la directory per un flusso di immagini
			// Get most recent image path logic here...
		}

		if (im.empty()) {
			continue;
		}

		// Effettua la predizione
		double confidence = 0.0;
		int prediction = -1;
		frw.Predict(im, prediction, confidence);

		std::cout << "Predicted: " << prediction << ", " << frw.GetLabelName(prediction) << " (" << confidence << ")" << std::endl;

		if (confidence < threshold && frw.GetLabelName(prediction) == username) {
			return PAM_SUCCESS;
		}
	}

	std::cout << "Timeout on face authentication..." << std::endl;
	return PAM_AUTH_ERR;
}
