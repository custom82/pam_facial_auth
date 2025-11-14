#include "../include/libfacialauth.h"

#include <getopt.h>
#include <iostream>

static void print_usage(const char *prog) {
	std::cerr
	<< "Usage: " << prog << " -u <user> -m <model_path> [options]\n\n"
	<< "Options:\n"
	<< "  -u, --user <user>        Utente da verificare (obbligatorio)\n"
	<< "  -m, --model <path>       File modello XML (default: basedir/models/<user>.xml)\n"
	<< "  -c, --config <file>      File di configurazione (default: /etc/security/pam_facial.conf)\n"
	<< "  -d, --device <device>    Dispositivo webcam (es. /dev/video0)\n"
	<< "      --threshold <value>  Soglia di confidenza per il match (default: 80.0)\n"
	<< "  -v, --verbose            Modalità verbosa\n"
	<< "      --nogui              Disabilita la GUI (solo console)\n"
	<< "  -h, --help               Mostra questo messaggio\n";
}

int main(int argc, char **argv) {
	std::string user;
	std::string model_path;
	std::string config_path = "/etc/security/pam_facial.conf";
	bool verbose = false;

	FacialAuthConfig cfg;

	static struct option long_opts[] = {
		{"user",      required_argument, 0, 'u'},
		{"model",     required_argument, 0, 'm'},
		{"config",    required_argument, 0, 'c'},
		{"device",    required_argument, 0, 'd'},
		{"threshold", required_argument, 0, 1},
		{"nogui",     no_argument,       0, 2},
		{"verbose",   no_argument,       0, 'v'},
		{"help",      no_argument,       0, 'h'},
		{0,0,0,0}
	};

	int opt, idx;
	while ((opt = getopt_long(argc, argv, "u:m:c:d:vh", long_opts, &idx)) != -1) {
		switch (opt) {
			case 'u': user = optarg; break;
			case 'm': model_path = optarg; break;
			case 'c': config_path = optarg; break;
			case 'd': cfg.device = optarg; break;
			case 'v': verbose = true; break;
			case 'h': print_usage(argv[0]); return 0;
			case 1:  cfg.threshold = std::stod(optarg); break;
			case 2:  cfg.nogui = true; break;
			default:
				print_usage(argv[0]);
				return 1;
		}
	}

	if (user.empty()) {
		std::cerr << "Errore: devi specificare l'utente con -u <user>\n";
		print_usage(argv[0]);
		return 1;
	}

	std::string log;
	read_kv_config(config_path, cfg, &log);

	// Use the default model path if not provided
	if (model_path.empty()) {
		model_path = fa_user_model_path(cfg, user);
	}

	double best_conf;
	int best_label;
	bool ok = fa_test_user(user, cfg, model_path, best_conf, best_label, log);

	if (verbose || cfg.debug) {
		std::cerr << log;
		std::cerr << "Best label: " << best_label << "  conf: " << best_conf
		<< "  threshold: " << cfg.threshold << "\n";
	}

	return ok ? 0 : 1;
}
