#include "../include/libfacialauth.h"
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

int main(int argc, char **argv)
{
	std::string user;
	std::string config_path = "/etc/security/pam_facial.conf";
	bool force = false;
	std::string input_dir;
	std::string output_model;
	std::string method = "lbph";
	bool debug = false;

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
		else if ((arg == "-i" || arg == "--input") && i + 1 < argc)
			input_dir = argv[++i];
		else if ((arg == "-o" || arg == "--output") && i + 1 < argc)
			output_model = argv[++i];
		else if ((arg == "-m" || arg == "--method") && i + 1 < argc)
			method = argv[++i];
		else if (arg == "-f" || arg == "--force")
			force = true;
		else if (arg == "-v" || arg == "--debug")
			debug = true;
		else if (arg == "-h" || arg == "--help")
		{
			std::cout <<
			"Usage: facial_training [options]\n"
			"Options:\n"
			"  -u, --user <name>         Username to train\n"
			"  -i, --input <dir>         Input directory with training images\n"
			"  -o, --output <file>       Output model file path\n"
			"  -c, --config <file>       Config file path (default: /etc/security/pam_facial.conf)\n"
			"  -m, --method <lbph>       Training method (default: lbph)\n"
			"  -f, --force               Force overwrite of model\n"
			"  -v, --debug               Enable debug messages\n"
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
	// Read configuration file
	// --------------------------------------------------
	FacialAuthConfig cfg;
	cfg.debug = debug;

	std::string log;
	read_kv_config(config_path, cfg, &log);
	if (cfg.debug)
		std::cerr << log;

	if (cfg.model_path.empty())
	{
		cfg.model_path = fs::path(cfg.basedir) / "models";
	}

	ensure_dirs(cfg.model_path);
	std::string model_path = output_model.empty()
	? fa_user_model_path(cfg, user)
	: output_model;

	// --------------------------------------------------
	// Run training
	// --------------------------------------------------
	std::string effective_input_dir = input_dir.empty()
	? fa_user_image_dir(cfg, user)
	: input_dir;

	std::string train_log;
	bool ok = fa_train_user(user, cfg, method, effective_input_dir, model_path, force, train_log);

	std::cout << train_log;

	if (!ok)
	{
		std::cerr << "Training failed for user: " << user << "\n";
		return 2;
	}

	std::cout << "✅ Model successfully trained: " << model_path << std::endl;
	return 0;
}
