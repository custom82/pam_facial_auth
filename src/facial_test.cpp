#include "../include/libfacialauth.h"
#include <getopt.h>
#include <iostream>

static void print_usage(const char *prog) {
	std::cerr << "Usage: " << prog << " -u <user> [options]\n\n"
	<< "Options:\n"
	<< "  -u, --user <user>        User name (required)\n"
	<< "  -m, --model <file>       Path to model file (default: basedir/models/<user>.xml)\n"
	<< "  -c, --config <file>      Config file (default: /etc/security/pam_facial.conf)\n"
	<< "  -n, --frames <num>       Number of frames to test (default: 5)\n"
	<< "  -t, --threshold <val>    Recognition threshold (default: 80.0)\n"
	<< "  -v, --verbose            Enable debug/verbose output\n"
	<< "  -g, --nogui              Disable GUI window\n"
	<< "  -h, --help               Show this help\n";
}

int main(int argc, char **argv) {
	std::string user;
	std::string model_path;
	std::string config_path = "/etc/security/pam_facial.conf";
	bool verbose = false;
	bool nogui = false;
	double threshold = 80.0;
	int frames = 5;

	FacialAuthConfig cfg;

	static struct option long_opts[] = {
		{"user", required_argument, 0, 'u'},
		{"model", required_argument, 0, 'm'},
		{"config", required_argument, 0, 'c'},
		{"frames", required_argument, 0, 'n'},
		{"threshold", required_argument, 0, 't'},
		{"verbose", no_argument, 0, 'v'},
		{"nogui", no_argument, 0, 'g'},
		{"help", no_argument, 0, 'h'},
		{0,0,0,0}
	};

	int opt, idx;
	while ((opt = getopt_long(argc, argv, "u:m:c:n:t:vgh", long_opts, &idx)) != -1) {
		switch (opt) {
			case 'u': user = optarg; break;
			case 'm': model_path = optarg; break;
			case 'c': config_path = optarg; break;
			case 'n': frames = atoi(optarg); break;
			case 't': threshold = atof(optarg); break;
			case 'v': verbose = true; break;
			case 'g': nogui = true; break;
			case 'h': print_usage(argv[0]); return 0;
			default: print_usage(argv[0]); return 1;
		}
	}

	if (user.empty()) {
		std::cerr << "Error: user is required.\n";
		print_usage(argv[0]);
		return 1;
	}

	read_kv_config(config_path, cfg);
	cfg.frames = frames;
	cfg.threshold = threshold;
	cfg.nogui = nogui;
	if (verbose) cfg.debug = true;

	if (model_path.empty()) {
		model_path = fa_user_model_path(cfg, user);
	}

	double best_conf = 9999.0;
	int best_label = -1;
	std::string log;

	bool ok = fa_test_user(user, cfg, model_path, best_conf, best_label, log);

	// log live già visibile grazie a append_and_emit
	std::cerr << log;

	if (ok)
		std::cout << "✅ Authentication success (confidence = " << best_conf << ")\n";
	else
		std::cout << "❌ Authentication failed (best confidence = " << best_conf << ")\n";

	return ok ? 0 : 1;
}
