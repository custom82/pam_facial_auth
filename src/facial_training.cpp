#include "FacialAuth.h"
#include <iostream>

int main(int argc, char **argv) {
	FacialAuthConfig cfg;
	FacialAuth::load_config(cfg, (const char**)argv+1, argc-1, nullptr);
	if (argc < 2) {
		std::cerr << "Usage: facial_training <user> [key=val ...]\n";
		return 2;
	}
	std::string user = argv[1];
	if (FacialAuth::auto_train_from_camera(cfg, user)) {
		std::cout << "Training OK for user " << user << "\n";
		return 0;
	}
	std::cerr << "Training FAILED for user " << user << "\n";
	return 1;
}
