#include "../include/libfacialauth.h"

#include <getopt.h>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

static void usage(const char *prog)
{
	std::cerr <<
	"Usage: " << prog << " [options]\n"
	"  -u, --user USER             User name\n"
	"  -m, --method METHOD         lbph|eigen|fisher|dnn [default: lbph]\n"
	"  -i, --input-dir DIR         Training images dir\n"
	"  -o, --output-model FILE     Output model XML\n"
	"  -c, --config FILE           Config file (default " FACIALAUTH_CONFIG_DEFAULT ")\n"
	"  -f, --force                 Force overwrite\n"
	"      --dnn-type T            caffe|tensorflow|onnx|openvino|tflite|torch\n"
	"      --dnn-model PATH        DNN model path\n"
	"      --dnn-proto PATH        DNN proto/config path\n"
	"      --dnn-device DEV        cpu|cuda|opencl|openvino\n"
	"      --dnn-threshold VAL     DNN threshold [0-1], default 0.6\n"
	"      --dnn-profile NAME      DNN profile (fast,sface,lresnet100,openface,\n"
	"                              yunet,emotion,keypoints,det_uint8,det_caffe,\n"
	"                              det_fp16,mp_landmark,mp_face,mp_blend)\n"
	"      --debug                 Enable debug logging\n";
}

int main(int argc, char *argv[])
{
	FacialAuthConfig cfg;

	std::string user;
	std::string input_dir;
	std::string method = "lbph";
	std::string log;
	std::string output_model;
	std::string config_path = FACIALAUTH_CONFIG_DEFAULT;
	std::string dnn_profile_cli;
	bool force = false;

	static struct option long_opts[] = {
		{"user",          required_argument, nullptr, 'u'},
		{"method",        required_argument, nullptr, 'm'},
		{"input-dir",     required_argument, nullptr, 'i'},
		{"output-model",  required_argument, nullptr, 'o'},
		{"config",        required_argument, nullptr, 'c'},
		{"force",         no_argument,       nullptr, 'f'},
		{"dnn-type",      required_argument, nullptr,  1 },
		{"dnn-model",     required_argument, nullptr,  2 },
		{"dnn-proto",     required_argument, nullptr,  3 },
		{"dnn-device",    required_argument, nullptr,  4 },
		{"dnn-threshold", required_argument, nullptr,  5 },
		{"debug",         no_argument,       nullptr,  6 },
		{"dnn-profile",   required_argument, nullptr,  7 },
		{nullptr,         0,                 nullptr,  0 }
	};

	int opt, idx;
	while ((opt = getopt_long(argc, argv, "u:m:i:o:c:f", long_opts, &idx)) != -1) {
		switch (opt) {
			case 'u':
				user = optarg;
				break;
			case 'm':
				method = optarg;
				break;
			case 'i':
				input_dir = optarg;
				break;
			case 'o':
				output_model = optarg;
				break;
			case 'c':
				config_path = optarg;
				break;
			case 'f':
				force = true;
				cfg.force_overwrite = true;
				break;
			case 1:
				cfg.dnn_type = optarg;
				break;
			case 2:
				cfg.dnn_model_path = optarg;
				break;
			case 3:
				cfg.dnn_proto_path = optarg;
				break;
			case 4:
				cfg.dnn_device = optarg;
				break;
			case 5:
				cfg.dnn_threshold = std::stod(optarg);
				break;
			case 6:
				cfg.debug = true;
				break;
			case 7:
				dnn_profile_cli = optarg;
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

	// Carica config da file (sovrascrive default/valori iniziali di cfg)
	fa_load_config(config_path, cfg, log);

	// Se il metodo è DNN, seleziona il profilo:
	//  - se --dnn-profile è specificato, ha priorità
	//  - altrimenti usa cfg.dnn_profile dal file di configurazione
	{
		std::string mt = method;
		for (auto &c : mt) c = static_cast<char>(std::tolower((unsigned char)c));
		if (mt == "dnn") {
			std::string profile_to_use;
			if (!dnn_profile_cli.empty()) {
				profile_to_use = dnn_profile_cli;
			} else {
				profile_to_use = cfg.dnn_profile;
			}

			if (!profile_to_use.empty()) {
				if (!fa_select_dnn_profile(cfg, profile_to_use, log)) {
					std::cerr << "[ERROR] Unknown DNN profile: " << profile_to_use << "\n";
					return 1;
				}
			}
		}
	}

	// Se input_dir non specificata, useremo quella di default dentro fa_train_user

	if (!fa_train_user(user, cfg, method, input_dir, output_model, force, log)) {
		std::cerr << "[ERROR] Training failed for user " << user << "\n";
		return 1;
	}

	std::cout << "[INFO] Training completed for user " << user << "\n";
	return 0;
}
