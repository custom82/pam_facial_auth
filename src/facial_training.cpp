#include "../include/libfacialauth.h"
#include <getopt.h>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

int main(int argc, char *argv[]) {

	FacialAuthConfig cfg;

	std::string user;
	std::string input_dir;
	std::string method = "lbph";
	std::string log;

	std::string config_path = FACIALAUTH_CONFIG_DEFAULT;
	bool force = false;

	static struct option long_opts[] = {
		{"user",   required_argument, 0, 'u'},
		{"input",  required_argument, 0, 'i'},
		{"method", required_argument, 0, 'm'},
		{"config", required_argument, 0, 'c'},
		{"force",  no_argument,       0, 'f'},
		{"debug",  no_argument,       0, 'v'},
		{0,0,0,0}
	};

	int opt, idx = 0;

	while ((opt = getopt_long(argc, argv, "u:i:m:c:fv", long_opts, &idx)) != -1) {
		switch (opt) {
			case 'u': user = optarg; break;
			case 'i': input_dir = optarg; break;
			case 'm': method = optarg; break;
			case 'c': config_path = optarg; break;
			case 'f': force = true; break;
			case 'v': cfg.debug = true; break;
			default:
				std::cerr << "Unknown option\n";
				return 1;
		}
	}

	if (user.empty()) {
		std::cerr << "[ERROR] --user is required\n";
		return 1;
	}

	if (input_dir.empty()) {
		std::cerr << "[ERROR] --input is required\n";
		return 1;
	}

	// Load config file
	read_kv_config(config_path, cfg, &log);

	// Determine model path
	std::string model_path = fa_user_model_path(cfg, user);

	if (fs::exists(model_path) && !force) {
		std::cerr << "[ERROR] Model already exists: "
		<< model_path << "\nUse --force to overwrite.\n";
		return 1;
	}

	if (fs::exists(model_path) && force) {
		std::cout << "[INFO] Overwriting model: " << model_path << "\n";
	}

	if (!fs::exists(input_dir)) {
		std::cerr << "[ERROR] Input directory does not exist: "
		<< input_dir << "\n";
		return 1;
	}

	int valid_images = 0;

	for (auto &entry : fs::directory_iterator(input_dir)) {
		if (entry.is_regular_file()) {
			if (fa_is_valid_image(entry.path().string()))
				valid_images++;
		}
	}

	if (valid_images == 0) {
		std::cerr << "[ERROR] No valid images found in "
		<< input_dir << "\n";
		return 1;
	}

	if (!fa_train_user(user, cfg, method, input_dir, model_path, force, log)) {
		std::cerr << "[ERROR] Training failed\n";
		return 1;
	}

	std::cout << "✅ Model successfully trained: " << model_path << "\n";
	return 0;
}
