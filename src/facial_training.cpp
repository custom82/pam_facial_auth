#include "../include/libfacialauth.h"
#include <getopt.h>
#include <iostream>

static void print_usage(const char *prog) {
	std::cerr << "Usage: " << prog << " -u <user> [options]\n\n"
	<< "Options:\n"
	<< "  -u, --user <user>        User name (required)\n"
	<< "  -i, --input <dir>        Directory of training images\n"
	<< "  -m, --model <file>       Output model path (default: basedir/models/<user>.xml)\n"
	<< "  -c, --config <file>      Config file (default: /etc/security/pam_facial.conf)\n"
	<< "  -f, --force              Overwrite existing model\n"
	<< "  -v, --verbose            Verbose/debug output\n"
	<< "  -h, --help               Show this message\n";
}

int main(int argc, char **argv) {
	std::string user;
	std::string input_dir;
	std::string model_path;
	std::string config_path = "/etc/security/pam_facial.conf";
	bool verbose = false;
	bool force = false;

	FacialAuthConfig cfg;

	static struct option long_opts[] = {
		{"user", required_argument, 0, 'u'},
		{"input", required_argument, 0, 'i'},
		{"model", required_argument, 0, 'm'},
		{"config", required_argument, 0, 'c'},
		{"force", no_argument, 0, 'f'},
		{"verbose", no_argument, 0, 'v'},
		{"help", no_argument, 0, 'h'},
		{0,0,0,0}
	};

	int opt, idx;
	while ((opt = getopt_long(argc, argv, "u:i:m:c:fvh", long_opts, &idx)) != -1) {
		switch (opt) {
			case 'u': user = optarg; break;
			case 'i': input_dir = optarg; break;
			case 'm': model_path = optarg; break;
			case 'c': config_path = optarg; break;
			case 'f': force = true; break;
			case 'v': verbose = true; break;
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
	if (verbose) cfg.debug = true;

	// Determina il path del modello
	if (model_path.empty()) {
		model_path = fa_user_model_path(cfg, user);
	}

	std::string log;
	bool ok = fa_train_user(user, cfg, "lbph", input_dir, model_path, force, log);

	std::cerr << log;
	return ok ? 0 : 1;
}
