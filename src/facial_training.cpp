#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <filesystem>
#include <vector>
#include <string>

namespace fs = std::filesystem;

// ======== COLORI ANSI ========
#define COLOR_RESET   "\033[0m"
#define COLOR_RED     "\033[31m"
#define COLOR_GREEN   "\033[32m"
#define COLOR_YELLOW  "\033[33m"
#define COLOR_CYAN    "\033[36m"

// ======== FLAG GLOBALI ========
bool verbose = false;
bool force_overwrite = false;

// ======== FUNZIONI DI LOG ========
void log_info(const std::string& msg) {
	std::cout << COLOR_CYAN << "[INFO] " << COLOR_RESET << msg << std::endl;
}

void log_success(const std::string& msg) {
	std::cout << COLOR_GREEN << "[OK] " << COLOR_RESET << msg << std::endl;
}

void log_debug(const std::string& msg) {
	if (verbose)
		std::cout << COLOR_YELLOW << "[DEBUG] " << COLOR_RESET << msg << std::endl;
}

void log_error(const std::string& msg) {
	std::cerr << COLOR_RED << "[ERROR] " << COLOR_RESET << msg << std::endl;
}

// ======== FUNZIONE HELP ========
void print_usage() {
	std::cout << COLOR_CYAN
	<< "Usage: facial_training -u <user> -m <method> <training_data_directory> [options]\n\n"
	<< COLOR_RESET
	<< "Options:\n"
	<< "  -u, --user <name>           Specify the username to train the model for\n"
	<< "  -m, --method <type>         Specify the training method (lbph, eigen, fisher)\n"
	<< "  -o, --output <file>         Path to save the trained model (XML)\n"
	<< "  -f, --force                 Force overwrite of existing model file\n"
	<< "  -v, --verbose               Enable detailed output\n"
	<< "  -h, --help                  Show this help message\n\n"
	<< "Examples:\n"
	<< "  facial_training -u custom -m lbph /etc/pam_facial_auth/ -o /etc/pam_facial_auth/custom/models/custom.xml\n"
	<< "  facial_training -u alice -m eigen /etc/pam_facial_auth/ --force --verbose\n"
	<< std::endl;
}

// ======== MAIN ========
int main(int argc, char** argv) {
	if (argc < 4) {
		print_usage();
		return 1;
	}

	std::string user;
	std::string method;
	std::string training_data_directory;
	std::string output_model;

	// --- Parsing parametri ---
	for (int i = 1; i < argc; ++i) {
		std::string arg = argv[i];

		if ((arg == "-u" || arg == "--user") && i + 1 < argc)
			user = argv[++i];
		else if ((arg == "-m" || arg == "--method") && i + 1 < argc)
			method = argv[++i];
		else if ((arg == "-o" || arg == "--output") && i + 1 < argc)
			output_model = argv[++i];
		else if ((arg == "-f" || arg == "--force"))
			force_overwrite = true;
		else if ((arg == "-v" || arg == "--verbose"))
			verbose = true;
		else if ((arg == "-h" || arg == "--help")) {
			print_usage();
			return 0;
		}
		else if (arg[0] != '-')
			training_data_directory = arg;
	}

	if (user.empty() || method.empty() || training_data_directory.empty()) {
		print_usage();
		return 1;
	}

	// Percorso directory immagini dell'utente
	fs::path user_path = fs::path(training_data_directory) / user;
	if (!fs::exists(user_path) || !fs::is_directory(user_path)) {
		log_error("User directory not found: " + user_path.string());
		return 1;
	}

	log_info("User: " + user);
	log_info("Method: " + method);
	log_info("Training directory: " + user_path.string());

	// Output file
	if (output_model.empty()) {
		output_model = (user_path / "models" / (user + "_model.xml")).string();
		log_info("No output specified, using default: " + output_model);
	}

	fs::path output_path(output_model);
	fs::create_directories(output_path.parent_path());

	// Se esiste già
	if (fs::exists(output_path) && !force_overwrite) {
		log_error("Output model already exists, use -f or --force to overwrite.");
		return 1;
	}
	if (force_overwrite)
		log_debug("Forcing overwrite of existing model file.");

	// --- Carica immagini ---
	std::vector<cv::Mat> images;
	std::vector<int> labels;
	int label = 0;

	log_debug("Scanning directory: " + user_path.string());
	for (const auto& entry : fs::directory_iterator(user_path)) {
		if (!entry.is_regular_file()) continue;

		std::string file_path = entry.path().string();
		std::string ext = entry.path().extension().string();
		if (ext != ".jpg" && ext != ".jpeg" && ext != ".png")
			continue;

		cv::Mat img = cv::imread(file_path, cv::IMREAD_GRAYSCALE);
		if (img.empty()) {
			log_error("Failed to load image: " + file_path);
			continue;
		}

		images.push_back(img);
		labels.push_back(label);
		log_debug("Loaded image: " + file_path);
	}

	if (images.empty()) {
		log_error("No training images found in " + user_path.string());
		return 1;
	}

	// --- Selezione modello ---
	cv::Ptr<cv::face::FaceRecognizer> model;
	if (method == "lbph")
		model = cv::face::LBPHFaceRecognizer::create();
	else if (method == "eigen")
		model = cv::face::EigenFaceRecognizer::create();
	else if (method == "fisher")
		model = cv::face::FisherFaceRecognizer::create();
	else {
		log_error("Invalid method: " + method);
		return 1;
	}

	log_info("Training model with " + std::to_string(images.size()) + " images...");
	model->train(images, labels);

	model->save(output_model);
	log_success("Model saved successfully to: " + output_model);

	return 0;
}
