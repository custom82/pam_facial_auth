#include "../include/libfacialauth.h"

#include <getopt.h>
#include <iostream>

static void usage(const char *prog)
{
	std::cerr <<
	"Usage: " << prog << " [options]\n"
	"  -u, --user USER             User name\n"
	"  -m, --model FILE            Model XML (default: basedir/models/<user>.xml)\n"
	"  -c, --config FILE           Config file (default " FACIALAUTH_CONFIG_DEFAULT ")\n"
	"      --frames N              Number of frames\n"
	"      --debug                 Enable debug logging\n";
}

int main(int argc, char *argv[])
{
	FacialAuthConfig cfg;
	std::string config_path = FACIALAUTH_CONFIG_DEFAULT;
	std::string user;
	std::string model_path;
	std::string log;

	static struct option long_opts[] = {
		{"user",   required_argument, nullptr, 'u'},
		{"model",  required_argument, nullptr, 'm'},
		{"config", required_argument, nullptr, 'c'},
		{"frames", required_argument, nullptr,  1 },
		{"debug",  no_argument,       nullptr,  2 },
		{nullptr,  0,                 nullptr,  0 }
	};

	int opt, idx;
	while ((opt = getopt_long(argc, argv, "u:m:c:", long_opts, &idx)) != -1) {
		switch (opt) {
			case 'u':
				user = optarg;
				break;
			case 'm':
				model_path = optarg;
				break;
			case 'c':
				config_path = optarg;
				break;
			case 1:
				cfg.frames = std::stoi(optarg);
				break;
			case 2:
				cfg.debug = true;
				break;
			default:
				usage(argv[0]);
				return 1;
		}
	}

	if (user.empty()) {
		usage(argv[0]);
		return 1;
	}

	fa_load_config(config_path, cfg, log);

	double best_conf = 0.0;
	int best_label   = -1;

	bool ok = fa_test_user(user, cfg, model_path, best_conf, best_label, log);

	std::cout << "Result: " << (ok ? "SUCCESS" : "FAIL") << "\n"
	<< "  best_conf = " << best_conf << "\n"
	<< "  best_label = " << best_label << "\n";

	return ok ? 0 : 1;
}
