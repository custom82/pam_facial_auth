#include "../include/libfacialauth.h"

#include <getopt.h>
#include <iostream>

static void print_usage(const char *prog) {
	std::cerr
	<< "Usage: " << prog << " -u <user> -m <method> <training_data_directory> [options]\n\n"
	<< "Options:\n"
	<< "  -u, --user <name>           Specify the username to train the model for\n"
	<< "  -m, --method <type>         Training method (lbph)\n"
	<< "  -o, --output <file>         Path to save the trained model (XML)\n"
	<< "  -c, --config <file>         Config file (default: /etc/security/pam_facial.conf)\n"
	<< "  -f, --force                 Force overwrite of existing model file\n"
	<< "  -v, --verbose               Enable detailed output\n"
	<< "  -h, --help                  Show this help message\n\n"
	<< "Examples:\n"
	<< "  " << prog << " -u custom -m lbph /etc/pam_facial_auth/images/custom -o /etc/pam_facial_auth/models/custom.xml\n";
}

int main(int argc, char **argv) {
	std::string user;
	std::string method;
	std::string output_model;
	std::string config_path = "/etc/security/pam_facial.conf";
	bool force = false;
	bool verbose = false;

	FacialAuthConfig cfg;

	static struct option long_opts[] = {
		{"user",    required_argument, 0, 'u'},
		{"method",  required_argument, 0, 'm'},
		{"output",  required_argument, 0, 'o'},
		{"config",  required_argument, 0, 'c'},
		{"force",   no_argument,       0, 'f'},
		{"verbose", no_argument,       0, 'v'},
		{"help",    no_argument,       0, 'h'},
		{0,0,0,0}
	};

	int opt, idx;
	while ((opt = getopt_long(argc, argv, "u:m:o:c:fvh", long_opts, &idx)) != -1) {
		switch (opt) {
			case 'u': user = optarg; break;
			case 'm': method = optarg; break;
			case 'o': output_model = optarg; break;
			case 'c': config_path = optarg; break;
			case 'f': force = true; break;
			case 'v': verbose = true; break;
			case 'h': print_usage(argv[0]); return 0;
			default:
				print_usage(argv[0]);
				return 1;
		}
	}

	if (user.empty() || method.empty()) {
		std::cerr << "Error: user and method are mandatory\n";
		print_usage(argv[0]);
		return 1;
	}

	std::string train_dir;
	if (optind < argc) {
		train_dir = argv[optind];
	}

	std::string log;
	read_kv_config(config_path, cfg, &log);

	if (output_model.empty()) {
		output_model = fa_user_model_path(cfg, user);
	}

	bool ok = fa_train_user(user, cfg, method, train_dir, output_model, force, log);

	if (verbose || cfg.debug) std::cerr << log;
	return ok ? 0 : 1;
}
