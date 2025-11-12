#include "FacialAuth.h"
#include <iostream>

int main(int argc, char **argv) {
	FacialAuthConfig cfg;
	FacialAuth::load_config(cfg, (const char**)argv+1, argc-1, nullptr);
	if (argc < 2) {
		std::cerr << "Usage: facial_test <user> [key=val ...]\n";
		return 2;
	}
	std::string user = argv[1];
	double conf=0.0;
	bool ok = FacialAuth::recognize_loop(cfg, user, false, nullptr, conf);
	if (ok) {
		std::cout << "OK conf=" << conf << "\n";
		return 0;
	}
	std::cout << "FAIL\n";
	return 1;
}
