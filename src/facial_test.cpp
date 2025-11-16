#include "../include/libfacialauth.h"
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

int main(int argc, char **argv)
{
	std::string user;
	std::string config_path = "/etc/security/pam_facial.conf";
	std::string model_path;
	bool debug = false;
	bool nogui = false;
	bool verbose = false;

	// --------------------------------------------------
	// Parse CLI arguments
	// --------------------------------------------------
	for (int i = 1; i < argc; ++i)
	{
		std::string arg = argv[i];
		if ((arg == "-u" || arg == "--user") && i + 1 < argc)
			user = argv[++i];
		else if ((arg == "-c" || arg == "--config") && i + 1 < argc)
			config_path = argv[++i];
		else if ((arg == "-m" || arg == "--model") && i + 1 < argc)
			model_path = argv[++i];
		else if (arg == "-v" || arg == "--debug")
			debug = true;
		else if (arg == "-n" || arg == "--nogui")
			nogui = true;
		else if (arg == "-V" || arg == "--verbose")
			verbose = true;
		else if (arg == "-h" || arg == "--help")
		{
			std::cout <<
			"Usage: facial_test [options]\n"
			"Options:\n"
			"  -u, --user <name>         Username to test\n"
			"  -m, --model <path>        Model XML path (optional)\n"
			"  -c, --config <file>       Config file path (default: /etc/security/pam_facial.conf)\n"
			"  -n, --nogui               Disable GUI preview\n"
			"  -v, --debug               Enable debug logs\n"
			"  -V, --verbose             Verbose output\n"
			"  -h, --help                Show this help\n";
			return 0;
		}
	}

	if (user.empty())
	{
		std::cerr << "Error: missing --user <name>\n";
		return 1;
	}

	// --------------------------------------------------
	// Load configuration
	// --------------------------------------------------
	FacialAuthConfig cfg;
	cfg.debug = debug;
	cfg.nogui = nogui;

	std::string log;
	read_kv_config(config_path, cfg, &log);

	if (cfg.debug || verbose)
		std::cerr << log;

	if (model_path.empty())
		model_path = fa_user_model_path(cfg, user);

	// --------------------------------------------------
	// Run face test
	// --------------------------------------------------
	double best_conf = 0.0;
	int best_label = -1;
	std::string test_log;

	bool ok = fa_test_user(user, cfg, model_path, best_conf, best_label, test_log);

	std::cout << test_log;

	if (!ok)
	{
		std::cerr << "❌ Face recognition failed for user: " << user
		<< " (best confidence=" << best_conf << ")\n";
		return 2;
	}

	std::cout << "✅ Authentication success for user: " << user
	<< " (confidence=" << best_conf << ")\n";
	return 0;
}
