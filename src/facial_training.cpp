#include <iostream>
#include <string>
#include <vector>
#include <getopt.h>
#include <unistd.h>
#include <filesystem>

#include "../include/libfacialauth.h"

namespace fs = std::filesystem;

// -----------------------------------------------------
// HELP
// -----------------------------------------------------
void print_usage(const char *prog) {
	std::cout
	<< "Usage: " << prog << " -u <user> -m <method> <training_data_directory> [options]\n\n"
	<< "Options:\n"
	<< "  -u, --user <name>           Specify the username to train the model for\n"
	<< "  -m, --method <type>         Specify the training method (lbph, eigen, fisher)\n"
	<< "  -o, --output <file>         Path to save the trained model (XML)\n"
	<< "  -f, --force                 Force overwrite of existing model file\n"
	<< "  -v, --verbose               Enable detailed output\n"
	<< "  -h, --help                  Show this help message\n\n"
	<< "Examples:\n"
	<< "  " << prog << " -u custom -m lbph /etc/pam_facial_auth/ -o /etc/pam_facial_auth/custom/models/custom.xml\n"
	<< "  " << prog << " -u alice -m eigen /etc/pam_facial_auth/ --force --verbose\n";
}

// -----------------------------------------------------
// Carica immagini dal dataset
// -----------------------------------------------------
static bool load_training_data(
	const std::string &path,
	std::vector<cv::Mat> &images,
	std::vector<int> &labels,
	bool verbose)
{
	if (!fs::exists(path)) {
		std::cerr << "[ERR] Training directory does not exist: " << path << "\n";
		return false;
	}

	for (const auto &user_dir : fs::directory_iterator(path)) {
		if (!user_dir.is_directory()) continue;

		std::string dirname = user_dir.path().filename().string();
		if (dirname == "." || dirname == "..") continue;

		int label = std::hash<std::string>{}(dirname) & 0x7FFFFFFF;

		std::string imgdir = user_dir.path().string() + "/images";
		if (!fs::exists(imgdir)) continue;

		for (const auto &img : fs::directory_iterator(imgdir)) {
			if (!img.is_regular_file()) continue;

			cv::Mat m = cv::imread(img.path().string(), cv::IMREAD_GRAYSCALE);
			if (m.empty()) {
				std::cerr << "[WARN] Could not read " << img.path() << "\n";
				continue;
			}

			images.push_back(m);
			labels.push_back(label);

			if (verbose)
				std::cout << "[INFO] Loaded: " << img.path().string() << "\n";
		}
	}

	if (images.empty()) {
		std::cerr << "[ERR] No training images found.\n";
		return false;
	}

	return true;
}

// -----------------------------------------------------
// MAIN
// -----------------------------------------------------
int main(int argc, char **argv)
{
	std::string user;
	std::string method;
	std::string output_file;
	bool force = false;
	bool verbose = false;

	if (argc < 2) {
		print_usage(argv[0]);
		return 1;
	}

	static struct option long_opts[] = {
		{"user",    required_argument, nullptr, 'u'},
		{"method",  required_argument, nullptr, 'm'},
		{"output",  required_argument, nullptr, 'o'},
		{"force",   no_argument,       nullptr, 'f'},
		{"verbose", no_argument,       nullptr, 'v'},
		{"help",    no_argument,       nullptr, 'h'},
		{nullptr, 0, nullptr, 0}
	};

	int opt, longidx = 0;

	while ((opt = getopt_long(argc, argv, "u:m:o:fvh", long_opts, &longidx)) != -1) {
		switch (opt) {
			case 'u':
				user = optarg;
				break;
			case 'm':
				method = optarg;
				break;
			case 'o':
				output_file = optarg;
				break;
			case 'f':
				force = true;
				break;
			case 'v':
				verbose = true;
				break;
			case 'h':
				print_usage(argv[0]);
				return 0;
			default:
				print_usage(argv[0]);
				return 1;
		}
	}

	if (optind >= argc) {
		std::cerr << "[ERR] Missing training_data_directory\n";
		print_usage(argv[0]);
		return 1;
	}

	std::string train_dir = argv[optind];

	// -----------------------------------------------------
	// Validazioni
	// -----------------------------------------------------
	if (user.empty()) {
		std::cerr << "[ERR] Missing -u / --user\n";
		return 1;
	}

	if (method != "lbph" && method != "eigen" && method != "fisher") {
		std::cerr << "[ERR] Invalid method: " << method << "\n"
		<< "Valid: lbph, eigen, fisher\n";
		return 1;
	}

	if (output_file.empty()) {
		output_file = "/etc/pam_facial_auth/" + user + "/models/" + user + ".xml";
	}

	if (fs::exists(output_file) && !force) {
		std::cerr << "[ERR] Output model exists. Use --force to overwrite.\n";
		return 1;
	}

	ensure_dirs(fs::path(output_file).parent_path().string());

	// -----------------------------------------------------
	// Carica dataset immagini
	// -----------------------------------------------------
	std::vector<cv::Mat> images;
	std::vector<int> labels;

	if (!load_training_data(train_dir, images, labels, verbose)) {
		return 1;
	}

	// -----------------------------------------------------
	// Inizializza riconoscitore
	// -----------------------------------------------------
	FaceRecWrapper trainer("/etc/pam_facial_auth", user, method);

	if (verbose)
		std::cout << "[INFO] Training model: " << method << "\n";

	trainer.Train(images, labels);

	// -----------------------------------------------------
	// Salvataggio modello
	// -----------------------------------------------------
	trainer.Save(output_file);

	if (verbose)
		std::cout << "[INFO] Model saved to: " << output_file << "\n";

	return 0;
}
