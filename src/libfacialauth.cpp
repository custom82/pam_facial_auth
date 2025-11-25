#include "../include/libfacialauth.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <cstdarg>
#include <cctype>
#include <cstring>
#include <getopt.h>
#include <sys/stat.h>
#include <unistd.h>

namespace fs = std::filesystem;

// ==========================================================
// Utility
// ==========================================================

std::string trim(const std::string &s) {
	size_t b = s.find_first_not_of(" \t\r\n");
	if (b == std::string::npos) return "";
	size_t e = s.find_last_not_of(" \t\r\n");
	return s.substr(b, e - b + 1);
}

bool str_to_bool(const std::string &s, bool defval) {
	std::string t = trim(s);
	for (char &c : t)
		c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

	if (t == "1" || t == "true" || t == "yes" || t == "on")  return true;
	if (t == "0" || t == "false" || t == "no" || t == "off") return false;
	return defval;
}

std::string join_path(const std::string &a, const std::string &b) {
	if (a.empty()) return b;
	if (b.empty()) return a;
	if (a.back() == '/') return a + b;
	return a + "/" + b;
}

void sleep_ms(int ms) {
	if (ms > 0)
		usleep(static_cast<useconds_t>(ms) * 1000);
}

bool file_exists(const std::string &path) {
	struct stat st {};
	return (::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode));
}

void ensure_dirs(const std::string &path) {
	if (path.empty()) return;
	try {
		fs::create_directories(path);
	} catch (...) {}
}

// ==========================================================
// Logging utility
// ==========================================================

void log_tool(const FacialAuthConfig &cfg, const char *level, const char *fmt, ...)
{
	char buf[1024];

	va_list ap;
	va_start(ap, fmt);
	vsnprintf(buf, sizeof(buf), fmt, ap);
	va_end(ap);

	const char *lev = level ? level : "";
	std::string msg = "[" + std::string(lev) + "] " + buf + "\n";
	bool is_err = (std::strcmp(lev, "ERROR") == 0);

	if (cfg.debug || is_err)
		std::fwrite(msg.c_str(), 1, msg.size(), stderr);

	if (!cfg.log_file.empty()) {
		std::ofstream logf(cfg.log_file, std::ios::app);
		if (logf.is_open())
			logf << msg;
	}
}

// ==========================================================
// Config file parser
// ==========================================================

bool read_kv_config(const std::string &path,
					FacialAuthConfig &cfg,
					std::string *logbuf)
{
	std::ifstream in(path);
	if (!in.is_open()) {
		if (logbuf) *logbuf += "Could not open config: " + path + "\n";
		return false;
	}

	std::string line;
	while (std::getline(in, line)) {
		line = trim(line);
		if (line.empty() || line[0] == '#')
			continue;

		std::string key, val;
		size_t eq = line.find('=');

		if (eq != std::string::npos) {
			key = trim(line.substr(0, eq));
			val = trim(line.substr(eq + 1));
		} else {
			std::istringstream iss(line);
			if (!(iss >> key))
				continue;
			std::getline(iss, val);
			val = trim(val);
		}

		try {
			if (key == "basedir") cfg.basedir = val;
			else if (key == "device") cfg.device = val;
			else if (key == "width")  cfg.width  = std::max(64, std::stoi(val));
			else if (key == "height") cfg.height = std::max(64, std::stoi(val));
			else if (key == "frames") cfg.frames = std::max(1, std::stoi(val));
			else if (key == "sleep_ms") cfg.sleep_ms = std::max(0, std::stoi(val));

			else if (key == "threshold")       cfg.threshold       = std::stod(val);
			else if (key == "lbph_threshold")  cfg.lbph_threshold  = std::stod(val);
			else if (key == "eigen_threshold") cfg.eigen_threshold = std::stod(val);
			else if (key == "fisher_threshold") cfg.fisher_threshold = std::stod(val);
			else if (key == "eigen_components") cfg.eigen_components = std::stoi(val);
			else if (key == "fisher_components") cfg.fisher_components = std::stoi(val);

			else if (key == "debug") cfg.debug = str_to_bool(val, cfg.debug);
			else if (key == "nogui") cfg.nogui = str_to_bool(val, cfg.nogui);
			else if (key == "fallback_device") cfg.fallback_device = str_to_bool(val, cfg.fallback_device);

			else if (key == "model_path")       cfg.model_path       = val;
			else if (key == "haar_cascade_path") cfg.haar_cascade_path = val;
			else if (key == "training_method")  cfg.training_method  = val;
			else if (key == "log_file")         cfg.log_file         = val;
			else if (key == "force_overwrite")  cfg.force_overwrite  = str_to_bool(val, false);
			else if (key == "ignore_failure")   cfg.ignore_failure   = str_to_bool(val, false);
		}
		catch (const std::exception &e) {
			if (logbuf)
				*logbuf += "Invalid line: " + line + " (" + e.what() + ")\n";
		}
	}

	return true;
}

// ==========================================================
// Path helpers
// ==========================================================

std::string fa_user_image_dir(const FacialAuthConfig &cfg, const std::string &user) {
	return join_path(join_path(cfg.basedir, "images"), user);
}

std::string fa_user_model_path(const FacialAuthConfig &cfg, const std::string &user) {
	return join_path(join_path(cfg.basedir, "models"), user + ".xml");
}

// ==========================================================
// Model type detection (from XML)
// ==========================================================

std::string fa_detect_model_type(const std::string &xmlPath)
{
	std::ifstream in(xmlPath);
	if (!in.is_open())
		return std::string();

	std::string line;
	while (std::getline(in, line)) {
		if (line.find("opencv_eigenfaces")  != std::string::npos ||
			line.find("EigenFaceRecognizer")  != std::string::npos ||
			line.find("eigenfaces")           != std::string::npos)
			return "eigen";

		if (line.find("opencv_fisherfaces") != std::string::npos ||
			line.find("FisherFaceRecognizer") != std::string::npos ||
			line.find("fisherfaces")          != std::string::npos)
			return "fisher";

		if (line.find("opencv_lbphfaces")   != std::string::npos ||
			line.find("LBPHFaceRecognizer") != std::string::npos ||
			line.find("lbph")               != std::string::npos)
			return "lbph";
	}

	return std::string();
}

// ==========================================================
// FaceRecWrapper IMPLEMENTATION
// ==========================================================

FaceRecWrapper::FaceRecWrapper(const std::string &modelType_)
: modelType(modelType_)
{
	CreateRecognizer();
}

bool FaceRecWrapper::CreateRecognizer()
{
	try {
		if (modelType == "eigen") {
			recognizer = cv::face::EigenFaceRecognizer::create();
		} else if (modelType == "fisher") {
			recognizer = cv::face::FisherFaceRecognizer::create();
		} else {
			// default LBPH
			recognizer = cv::face::LBPHFaceRecognizer::create();
			modelType  = "lbph";
		}

		return !recognizer.empty();
	} catch (...) {
		return false;
	}
}

bool FaceRecWrapper::InitCascade(const std::string &cascadePath)
{
	if (cascadePath.empty())
		return false;
	return faceCascade.load(cascadePath);
}

bool FaceRecWrapper::Load(const std::string &file)
{
	try {
		// Se possibile, deduci il tipo dal file XML
		std::string autoType = fa_detect_model_type(file);
		if (!autoType.empty()) {
			modelType = autoType;
			if (!CreateRecognizer())
				return false;
		} else {
			if (recognizer.empty() && !CreateRecognizer())
				return false;
		}

		recognizer->read(file);
		return true;
	} catch (...) {
		return false;
	}
}

bool FaceRecWrapper::Save(const std::string &file) const
{
	try {
		ensure_dirs(fs::path(file).parent_path().string());
		recognizer->write(file);
		return true;
	} catch (...) {
		return false;
	}
}

bool FaceRecWrapper::Train(const std::vector<cv::Mat> &images,
						   const std::vector<int>    &labels)
{
	if (images.empty() || labels.empty() || images.size() != labels.size())
		return false;

	if (recognizer.empty() && !const_cast<FaceRecWrapper*>(this)->CreateRecognizer())
		return false;

	try {
		recognizer->train(images, labels);
		return true;
	} catch (...) {
		return false;
	}
}

bool FaceRecWrapper::Predict(const cv::Mat &face,
							 int &pred,
							 double &conf) const
							 {
								 if (face.empty() || recognizer.empty())
									 return false;

								 try {
									 recognizer->predict(face, pred, conf);
									 return true;
								 } catch (...) {
									 return false;
								 }
							 }

							 bool FaceRecWrapper::DetectFace(const cv::Mat &frame,
															 cv::Rect &faceROI)
							 {
								 if (frame.empty())
									 return false;

								 if (faceCascade.empty())
									 return false; // InitCascade DEVE essere chiamata prima

									 cv::Mat gray;
								 cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
								 cv::equalizeHist(gray, gray);

								 std::vector<cv::Rect> faces;
								 faceCascade.detectMultiScale(gray, faces,
															  1.08, 3,
									  0, cv::Size(60, 60));

								 if (faces.empty()) return false;

								 faceROI = faces[0];
								 return true;
							 }

							 // ==========================================================
							 // Camera helper
							 // ==========================================================

							 static bool open_camera(const FacialAuthConfig &cfg, cv::VideoCapture &cap, std::string &dev_used)
							 {
								 dev_used = cfg.device;
								 cap.open(cfg.device);

								 if (!cap.isOpened() && cfg.fallback_device) {
									 cap.open("/dev/video1");
									 if (cap.isOpened())
										 dev_used = "/dev/video1";
								 }

								 if (!cap.isOpened()) return false;

								 cap.set(cv::CAP_PROP_FRAME_WIDTH,  cfg.width);
								 cap.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.height);
								 return true;
							 }

							 // ==========================================================
							 // CAPTURE IMAGES
							 // ==========================================================

							 bool fa_capture_images(const std::string &user,
													const FacialAuthConfig &cfg,
							   bool force,
							   std::string &logbuf,
							   const std::string &img_format)
							 {
								 std::string dev_used;
								 cv::VideoCapture cap;

								 if (!open_camera(cfg, cap, dev_used)) {
									 log_tool(cfg, "ERROR", "Cannot open camera %s", cfg.device.c_str());
		return false;
								 }

	log_tool(cfg, "INFO", "Camera opened on %s", dev_used.c_str());

	if (cfg.haar_cascade_path.empty() || !file_exists(cfg.haar_cascade_path)) {
		log_tool(cfg, "ERROR", "haar_cascade_path is missing or invalid");
		return false;
	}

	FaceRecWrapper rec("lbph");
	if (!rec.InitCascade(cfg.haar_cascade_path)) {
		log_tool(cfg, "ERROR", "Cannot load HAAR cascade: %s", cfg.haar_cascade_path.c_str());
		return false;
	}

	std::string img_dir = fa_user_image_dir(cfg, user);
	ensure_dirs(img_dir);

	// Determine start index based on existing files
	int start_idx = 0;

	if (!force && !cfg.force_overwrite) {
		for (auto &entry : fs::directory_iterator(img_dir)) {
			if (!entry.is_regular_file()) continue;

			std::string name = entry.path().filename().string();
			if (name.size() >= 8 && name.rfind("img_", 0) == 0) {
				try {
					int idx = std::stoi(name.substr(4, 3));
					if (idx > start_idx) start_idx = idx;
				} catch (...) {}
			}
		}
	}

	cv::Mat frame;
	int captured = 0;

	std::string fmt = img_format.empty() ? "jpg" : img_format;
	for (char &c : fmt)
		c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

								 while (captured < cfg.frames) {
									 cap >> frame;
									 if (frame.empty()) break;

									 cv::Rect roi;
									 if (!rec.DetectFace(frame, roi)) {
										 log_tool(cfg, "DEBUG", "No face detected");
										 continue;
									 }

									 cv::Mat face = frame(roi).clone();

									 cv::Mat gray;
									 cv::cvtColor(face, gray, cv::COLOR_BGR2GRAY);
									 cv::equalizeHist(gray, gray);

									 char buf[64];
									 std::snprintf(buf, sizeof(buf), "img_%03d.%s",
												   start_idx + captured + 1, fmt.c_str());

									 std::string out = join_path(img_dir, buf);

									 cv::imwrite(out, gray);
									 log_tool(cfg, "INFO", "Saved %s", out.c_str());

									 captured++;
									 sleep_ms(cfg.sleep_ms);
								 }

								 return captured > 0;
							 }

							 // ==========================================================
							 // TRAIN MODEL
							 // ==========================================================

							 bool fa_train_user(const std::string &user,
												const FacialAuthConfig &cfg,
						   const std::string &method,
						   const std::string &inputDir,
						   const std::string &outputModel,
						   bool /*force*/,
						   std::string &logbuf)
							 {
								 std::string m = method;
								 if (m.empty())
									 m = "lbph";

								 if (m != "lbph" && m != "eigen" && m != "fisher") {
									 log_tool(cfg, "ERROR", "Invalid method '%s'", m.c_str());
		return false;
								 }

	std::string train_dir = inputDir.empty()
							? fa_user_image_dir(cfg, user)
							: inputDir;

	if (!fs::exists(train_dir)) {
		log_tool(cfg, "ERROR", "Training dir missing: %s", train_dir.c_str());
		return false;
	}

	std::vector<cv::Mat> images;
	std::vector<int>    labels;

	auto has_suffix = [](const std::string &s, const char *suf) {
		const size_t ls = s.size();
		const size_t lf = std::strlen(suf);
		return (ls >= lf && s.compare(ls - lf, lf, suf) == 0);
	};

	int target_w = 200;
	int target_h = 200;

	for (auto &entry : fs::directory_iterator(train_dir)) {
		if (!entry.is_regular_file()) continue;

		std::string path = entry.path().string();

		std::string lower = path;
		for (char &c : lower)
			c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

		if (!(has_suffix(lower, ".png") ||
			  has_suffix(lower, ".jpg") ||
			  has_suffix(lower, ".jpeg")))
			continue;

		cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
		if (img.empty())
			continue;

		// Normalizzazione — ESSENZIALE per EIGEN/FISHER
		cv::resize(img, img, cv::Size(target_w, target_h));

		images.push_back(img);
		labels.push_back(0);

		if (m == "eigen" || m == "fisher") {
			cv::Mat flipped;
			cv::flip(img, flipped, 1);
			cv::resize(flipped, flipped, cv::Size(target_w, target_h));
			images.push_back(flipped);
			labels.push_back(1);
		}
	}

	if (images.empty()) {
		log_tool(cfg, "ERROR", "No valid images in training directory");
		return false;
	}

	FaceRecWrapper rec(m);

	if (!rec.Train(images, labels)) {
		log_tool(cfg, "ERROR", "Training failed (Eigen/Fisher require equal-sized images)");
		return false;
	}

	std::string model_out =
		outputModel.empty() ? fa_user_model_path(cfg, user) : outputModel;

	if (!rec.Save(model_out)) {
		log_tool(cfg, "ERROR", "Cannot save model to %s", model_out.c_str());
		return false;
	}

	if (m == "eigen" || m == "fisher") {
		log_tool(cfg, "INFO", "Eigen/Fisher: created synthetic labels (0/1)");
	}

	log_tool(cfg, "INFO", "Model saved to %s", model_out.c_str());
	return true;
							 }


// ==========================================================
// TEST USER
// ==========================================================

bool fa_test_user(const std::string &user,
				  const FacialAuthConfig &cfg,
				  const std::string &modelPath,
				  double &best_conf,
				  int &best_label,
				  std::string &logbuf)
{
	(void)logbuf;

	std::string model_file =
		modelPath.empty() ? fa_user_model_path(cfg, user) : modelPath;

		if (!file_exists(model_file)) {
			log_tool(cfg, "ERROR", "Model missing: %s", model_file.c_str());
		return false;
		}

	FaceRecWrapper rec("lbph");

	// Se possibile, sovrascrivi il tipo dal file stesso
	std::string model_type = fa_detect_model_type(model_file);
	if (!model_type.empty())
		rec = FaceRecWrapper(model_type);

	if (!rec.Load(model_file)) {
		log_tool(cfg, "ERROR", "Cannot load model: %s", model_file.c_str());
		return false;
	}

	// modello effettivo usato (dopo Load)
	model_type = rec.GetModelType();

	// ---------------------------
	// Seleziona threshold corretto
	// ---------------------------
	double threshold = 0.0;
	if (cfg.threshold > 0.0) {
		// override globale da config o CLI
		threshold = cfg.threshold;
	} else {
		threshold = cfg.lbph_threshold;
		if (model_type == "eigen")
			threshold = cfg.eigen_threshold;
		else if (model_type == "fisher")
			threshold = cfg.fisher_threshold;
	}

	// ---------------------------
	// Carica il riconoscitore corretto
	// ---------------------------

	if (cfg.haar_cascade_path.empty() || !file_exists(cfg.haar_cascade_path)) {
		log_tool(cfg, "ERROR", "haar_cascade_path is missing or invalid");
		return false;
	}

	if (!rec.InitCascade(cfg.haar_cascade_path)) {
		log_tool(cfg, "ERROR", "Cannot load HAAR cascade: %s", cfg.haar_cascade_path.c_str());
		return false;
	}

	std::string dev_used;
	cv::VideoCapture cap;

	if (!open_camera(cfg, cap, dev_used)) {
		log_tool(cfg, "ERROR", "Cannot open camera: %s", cfg.device.c_str());
		return false;
	}

	log_tool(cfg, "INFO", "Testing user %s on device %s",
			 user.c_str(), dev_used.c_str());

	best_conf  = 1e9;
	best_label = -1;

	cv::Mat frame;

	for (int i = 0; i < cfg.frames; i++) {
		cap >> frame;
		if (frame.empty()) continue;

		cv::Rect roi;
		if (!rec.DetectFace(frame, roi))
			continue;

		cv::Mat face = frame(roi).clone();

		cv::Mat gray;
		cv::cvtColor(face, gray, cv::COLOR_BGR2GRAY);
		cv::equalizeHist(gray, gray);

// NORMALIZZAZIONE PER EIGEN & FISHER: obbligatoria
		if (model_type == "eigen" || model_type == "fisher") {
			cv::resize(gray, gray, cv::Size(200, 200));
		}


		int    label = -1;
		double conf  = 1e9;

		if (!rec.Predict(gray, label, conf))
			continue;

		if (conf < best_conf) {
			best_conf  = conf;
			best_label = label;
		}

		if (conf <= threshold) {
			log_tool(cfg, "INFO", "Auth success (model=%s): conf=%.2f <= %.2f",
					 model_type.c_str(), conf, threshold);
			return true;
		}

		sleep_ms(cfg.sleep_ms);
	}

	log_tool(cfg, "WARN",
			 "Auth failed (model=%s): best_conf=%.2f threshold=%.2f",
			 model_type.c_str(), best_conf, threshold);

	return false;
}

// ==========================================================
// Maintenance
// ==========================================================

bool fa_clean_images(const FacialAuthConfig &cfg, const std::string &user)
{
	std::string dir = fa_user_image_dir(cfg, user);
	if (!fs::exists(dir))
		return true;

	try {
		for (auto &entry : fs::directory_iterator(dir)) {
			if (entry.is_regular_file())
				fs::remove(entry.path());
		}
		return true;
	} catch (...) {
		return false;
	}
}

bool fa_clean_model(const FacialAuthConfig &cfg, const std::string &user)
{
	std::string model = fa_user_model_path(cfg, user);
	if (!fs::exists(model))
		return true;

	try {
		fs::remove(model);
		return true;
	} catch (...) {
		return false;
	}
}

void fa_list_images(const FacialAuthConfig &cfg, const std::string &user)
{
	std::string dir = fa_user_image_dir(cfg, user);

	if (!fs::exists(dir)) {
		std::cout << "[INFO] No images for user " << user << "\n";
		return;
	}

	std::cout << "[INFO] Images for user " << user << ":\n";
	for (auto &entry : fs::directory_iterator(dir)) {
		if (entry.is_regular_file())
			std::cout << "  " << entry.path().filename().string() << "\n";
	}
}

// ==========================================================
// Root check
// ==========================================================

bool fa_check_root(const char *tool_name)
{
	if (geteuid() != 0) {
		std::cerr << "Error: " << (tool_name ? tool_name : "this program")
		<< " must be run as root.\n";
		return false;
	}
	return true;
}

// ----------------------------------------------------------
// facial_capture CLI
// ----------------------------------------------------------

static void print_facial_capture_usage(const char *p)
{
	std::cout
	<< "Usage: " << p << " -u USER [options]\n"
	<< "  -u, --user USER\n"
	<< "  -d, --device DEV\n"
	<< "  -w, --width N\n"
	<< "  -h, --height N\n"
	<< "  -n, --frames N\n"
	<< "  -s, --sleep MS\n"
	<< "  -f, --force\n"
	<< "  -g, --nogui\n"
	<< "      --clean            Remove all user images\n"
	<< "      --reset            Remove user model + images\n"
	<< "  -v, --debug\n"
	<< "  -c, --config FILE\n"
	<< "      --format EXT\n";
}

int facial_capture_cli_main(int argc, char *argv[])
{
	FacialAuthConfig cfg;

	std::string config_path = FACIALAUTH_CONFIG_DEFAULT;
	std::string user;
	std::string device_opt;
	std::string img_format = "jpg";

	int width_opt   = 0;
	int height_opt  = 0;
	int frames_opt  = 0;
	int sleep_opt   = -1;
	bool force      = false;
	bool debug_opt  = false;
	bool nogui_opt  = false;

	bool opt_clean  = false;
	bool opt_reset  = false;

	enum {
		OPT_FORMAT = 1000,
		OPT_HELP   = 1001,
		OPT_CLEAN  = 1002,
		OPT_RESET  = 1003
	};

	static struct option long_opts[] = {
		{"user",    required_argument, nullptr, 'u'},
		{"device",  required_argument, nullptr, 'd'},
		{"width",   required_argument, nullptr, 'w'},
		{"height",  required_argument, nullptr, 'h'},
		{"frames",  required_argument, nullptr, 'n'},
		{"sleep",   required_argument, nullptr, 's'},
		{"force",   no_argument,       nullptr, 'f'},
		{"nogui",   no_argument,       nullptr, 'g'},
		{"debug",   no_argument,       nullptr, 'v'},
		{"config",  required_argument, nullptr, 'c'},
		{"format",  required_argument, nullptr, OPT_FORMAT},
		{"help",    no_argument,       nullptr, OPT_HELP},
		{"clean",   no_argument,       nullptr, OPT_CLEAN},
		{"reset",   no_argument,       nullptr, OPT_RESET},
		{nullptr,   0,                 nullptr, 0}
	};

	int opt;
	int long_index = 0;

	while ((opt = getopt_long(argc, argv, "u:d:w:h:n:s:fgvc:", long_opts, &long_index)) != -1) {
		switch (opt) {

			case 'u': user = optarg ? optarg : ""; break;
			case 'd': device_opt = optarg ? optarg : ""; break;
			case 'w': width_opt = std::atoi(optarg); break;
			case 'h': height_opt = std::atoi(optarg); break;
			case 'n': frames_opt = std::atoi(optarg); break;
			case 's': sleep_opt = std::atoi(optarg); break;
			case 'f': force = true; break;
			case 'g': nogui_opt = true; break;
			case 'v': debug_opt = true; break;
			case 'c': if (optarg) config_path = optarg; break;
			case OPT_FORMAT: if (optarg) img_format = optarg; break;

			case OPT_CLEAN: opt_clean = true; break;
			case OPT_RESET: opt_reset = true; break;

			case OPT_HELP:
				print_facial_capture_usage(argv[0]);
				return 0;

			default:
				print_facial_capture_usage(argv[0]);
				return 1;
		}
	}

	if (user.empty()) {
		std::cerr << "[ERROR] --user is required\n";
		print_facial_capture_usage(argv[0]);
		return 1;
	}

	std::string logbuf;

	if (!read_kv_config(config_path, cfg, &logbuf)) {
		std::cerr << "[ERROR] Cannot read config file: " << config_path << "\n";
		if (!logbuf.empty()) std::cerr << logbuf;
		return 1;
	}

	if (debug_opt) cfg.debug = true;
	if (nogui_opt) cfg.nogui = true;
	if (!device_opt.empty()) cfg.device = device_opt;
	if (width_opt > 0)  cfg.width  = width_opt;
	if (height_opt > 0) cfg.height = height_opt;
	if (frames_opt > 0) cfg.frames = frames_opt;
	if (sleep_opt >= 0) cfg.sleep_ms = sleep_opt;

	if (!fa_check_root("facial_capture"))
		return 1;

	if (opt_clean) {
		if (!fa_clean_images(cfg, user)) {
			std::cerr << "[ERROR] Cannot clean images for user: " << user << "\n";
			return 1;
		}
		std::cout << "[INFO] Images cleaned for user: " << user << "\n";
		return 0;
	}

	if (opt_reset) {
		bool ok1 = fa_clean_images(cfg, user);
		bool ok2 = fa_clean_model(cfg, user);

		if (!ok1 || !ok2) {
			std::cerr << "[ERROR] Cannot reset data for user: " << user << "\n";
			return 1;
		}

		std::cout << "[INFO] Model and images reset for user: " << user << "\n";
		return 0;
	}

	std::cout << "[INFO] Starting capture for user: " << user << "\n";

	if (!fa_capture_images(user, cfg, force, logbuf, img_format)) {
		std::cerr << "[ERROR] Capture failed\n";
		if (!logbuf.empty()) std::cerr << logbuf;
		return 1;
	}

	std::cout << "[INFO] Capture completed\n";
	return 0;
}

// ----------------------------------------------------------
// facial_training CLI
// ----------------------------------------------------------

static void print_facial_training_usage(const char *p)
{
	std::cout <<
	"Usage: facial_training -u <user> -m <method> [options]\n"
	"\n"
	"Options:\n"
	"  -u, --user <name>           Specify the username to train the model for\n"
	"  -m, --method <type>         Specify the training method (lbph, eigen, fisher)\n"
	"  -i, --input <dir>           Base directory (overrides basedir from config)\n"
	"  -o, --output <file>         Path to save the trained model (XML)\n"
	"  -f, --force                 Force overwrite of existing model file\n"
	"  -v, --verbose               Enable detailed output\n"
	"  -h, --help                  Show this help message\n"
	"\n"
	"If -i is not specified, basedir is read from config and images are taken\n"
	"from  <basedir>/images/<user>\n";
}

int facial_training_cli_main(int argc, char *argv[])
{
	FacialAuthConfig cfg;
	std::string config_path = FACIALAUTH_CONFIG_DEFAULT;

	std::string user;
	std::string method;
	std::string input_dir;      // override basedir se usato
	std::string output_file;    // override modello
	std::string logbuf;

	bool force   = false;
	bool verbose = false;

	struct option long_opts[] = {
		{"user",    required_argument, nullptr, 'u'},
		{"method",  required_argument, nullptr, 'm'},
		{"input",   required_argument, nullptr, 'i'},
		{"output",  required_argument, nullptr, 'o'},
		{"force",   no_argument,       nullptr, 'f'},
		{"verbose", no_argument,       nullptr, 'v'},
		{"help",    no_argument,       nullptr, 'h'},
		{nullptr,0,nullptr,0}
	};

	int opt, idx = 0;
	while ((opt = getopt_long(argc, argv, "u:m:i:o:fvh", long_opts, &idx)) != -1) {
		switch (opt) {
			case 'u': user = optarg; break;
			case 'm': method = optarg; break;
			case 'i': input_dir = optarg; break;      // override basedir
			case 'o': output_file = optarg; break;    // override model path
			case 'f': force = true; break;
			case 'v': verbose = true; break;

			case 'h':
				print_facial_training_usage(argv[0]);
				return 0;

			default:
				print_facial_training_usage(argv[0]);
				return 1;
		}
	}

	// Load config (including default basedir!)
	if (!read_kv_config(config_path, cfg, &logbuf)) {
		std::cerr << "[ERROR] Cannot read config file: " << config_path << "\n";
		if (!logbuf.empty()) std::cerr << logbuf;
		return 1;
	}

	if (verbose) cfg.debug = true;

	if (user.empty()) {
		std::cerr << "ERROR: --user is required\n";
		return 1;
	}

	if (method.empty()) {
		std::cerr << "ERROR: --method is required\n";
		return 1;
	}

	if (method != "lbph" && method != "eigen" && method != "fisher") {
		std::cerr << "ERROR: Invalid method '" << method << "'\n";
		return 1;
	}

	// Override basedir only if -i was specified
	if (!input_dir.empty())
		cfg.basedir = input_dir;

	// Se modello non specificato: basedir/models/<user>.xml
	if (output_file.empty())
		output_file = fa_user_model_path(cfg, user);

	// Train usando immagini da <basedir>/images/<user>
	std::string train_dir = fa_user_image_dir(cfg, user);

	if (!fa_train_user(user, cfg, method, train_dir, output_file, force, logbuf)) {
		std::cerr << "Training failed\n";
		if (!logbuf.empty()) std::cerr << logbuf;
		return 1;
	}

	std::cout << "[OK] Model trained: " << output_file << "\n";
	return 0;
}

// ----------------------------------------------------------
// facial_test CLI
// ----------------------------------------------------------

// ----------------------------------------------------------
// facial_test
// ----------------------------------------------------------

static void print_facial_test_usage(const char *p)
{
	std::cout
	<< "Usage: " << p << " -u <user> [options]\n\n"
	<< "Options:\n"
	<< "  -u, --user <user>        Utente da verificare (obbligatorio)\n"
	<< "  -m, --model <path>       File modello XML (opzionale;\n"
	<< "                           default: <basedir>/models/<user>.xml)\n"
	<< "  -c, --config <file>      File di configurazione\n"
	<< "                           (default: " << FACIALAUTH_CONFIG_DEFAULT << ")\n"
	<< "  -d, --device <device>    Dispositivo webcam (es. /dev/video0)\n"
	<< "      --threshold <value>  Soglia di confidenza (override globale)\n"
	<< "  -v, --verbose            Modalità verbosa\n"
	<< "      --nogui              Disabilita la GUI\n"
	<< "  -h, --help               Mostra questo messaggio\n";
}

int facial_test_cli_main(int argc, char *argv[])
{
	FacialAuthConfig cfg;
	std::string config_path = FACIALAUTH_CONFIG_DEFAULT;

	std::string user;
	std::string model_path;
	std::string device_opt;
	std::string logbuf;

	bool verbose = false;
	bool nogui_opt = false;

	double threshold_override = 0.0;
	bool threshold_set = false;

	enum {
		OPT_THRESHOLD = 1000,
		OPT_NOGUI     = 1001
	};

	struct option long_opts[] = {
		{"user",      required_argument, nullptr, 'u'},
		{"model",     required_argument, nullptr, 'm'},
		{"config",    required_argument, nullptr, 'c'},
		{"device",    required_argument, nullptr, 'd'},
		{"threshold", required_argument, nullptr, OPT_THRESHOLD},
		{"verbose",   no_argument,       nullptr, 'v'},
		{"nogui",     no_argument,       nullptr, OPT_NOGUI},
		{"help",      no_argument,       nullptr, 'h'},
		{nullptr,0,nullptr,0}
	};

	int opt, idx = 0;
	while ((opt = getopt_long(argc, argv, "u:m:c:d:vh", long_opts, &idx)) != -1) {
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
			case 'd':
				device_opt = optarg;
				break;
			case 'v':
				verbose = true;
				break;

			case OPT_THRESHOLD:
				if (optarg) {
					threshold_override = std::strtod(optarg, nullptr);
					threshold_set = true;
				}
				break;

			case OPT_NOGUI:
				nogui_opt = true;
				break;

			case 'h':
				print_facial_test_usage(argv[0]);
				return 0;

			default:
				print_facial_test_usage(argv[0]);
				return 1;
		}
	}

	// user obbligatorio
	if (user.empty()) {
		std::cerr << "ERROR: --user is required\n";
		print_facial_test_usage(argv[0]);
		return 1;
	}

	if (!fa_check_root("facial_test"))
		return 1;

	// Carica config (qui prendiamo basedir & soglie)
	if (!read_kv_config(config_path, cfg, &logbuf)) {
		std::cerr << "[ERROR] Cannot read config file: " << config_path << "\n";
		if (!logbuf.empty()) std::cerr << logbuf;
		return 1;
	}

	if (verbose)
		cfg.debug = true;
	if (nogui_opt)
		cfg.nogui = true;
	if (!device_opt.empty())
		cfg.device = device_opt;

	// Se il modello non è stato specificato, deriviamolo da basedir
	if (model_path.empty())
		model_path = fa_user_model_path(cfg, user);

	// Threshold override globale (forza tutte le soglie)
	if (threshold_set) {
		cfg.threshold        = threshold_override;
		cfg.lbph_threshold   = threshold_override;
		cfg.eigen_threshold  = threshold_override;
		cfg.fisher_threshold = threshold_override;
	}

	double best_conf = 0.0;
	int best_label   = -1;

	bool ok = fa_test_user(user, cfg, model_path, best_conf, best_label, logbuf);
	if (!ok) {
		std::cerr << "Authentication FAILED (best_conf=" << best_conf
		<< ", threshold=" << cfg.threshold << ")\n";
		if (!logbuf.empty()) std::cerr << logbuf;
		return 2;
	}

	std::cout << "[OK] Authentication SUCCESS (conf=" << best_conf << ")\n";
	return 0;
}
