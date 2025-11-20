#include "../include/libfacialauth.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <cstdarg>
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
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
	for (char &c : t) c = static_cast<char>(::tolower(c));
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
	if (ms <= 0) return;
	usleep(static_cast<useconds_t>(ms) * 1000);
}

bool read_kv_config(const std::string &path, FacialAuthConfig &cfg, std::string *logbuf) {
	std::ifstream in(path);
	if (!in.is_open()) {
		if (logbuf) *logbuf += "Could not open config: " + path + "\n";
		return false;
	}

	std::string line;
	while (std::getline(in, line)) {
		line = trim(line);
		if (line.empty() || line[0] == '#') continue;

		std::string key, val;
		size_t eq = line.find('=');
		if (eq != std::string::npos) {
			key = trim(line.substr(0, eq));
			val = trim(line.substr(eq + 1));
		} else {
			std::istringstream iss(line);
			if (!(iss >> key)) continue;
			std::getline(iss, val);
			val = trim(val);
		}

		try {
			if (key == "basedir") cfg.basedir = val;
			else if (key == "device") cfg.device = val;
			else if (key == "width") cfg.width = std::max(64, std::stoi(val));
			else if (key == "height") cfg.height = std::max(64, std::stoi(val));
			else if (key == "threshold") cfg.threshold = std::stod(val);
			else if (key == "timeout") cfg.timeout = std::max(1, std::stoi(val));
			else if (key == "nogui") cfg.nogui = str_to_bool(val, cfg.nogui);
			else if (key == "debug") cfg.debug = str_to_bool(val, cfg.debug);
			else if (key == "frames") cfg.frames = std::max(1, std::stoi(val));
			else if (key == "fallback_device") cfg.fallback_device = str_to_bool(val, cfg.fallback_device);
			else if (key == "sleep_ms") cfg.sleep_ms = std::max(0, std::stoi(val));
			else if (key == "model_path") cfg.model_path = val;
			else if (key == "haar_cascade_path") cfg.haar_cascade_path = val;
			else if (key == "training_method") cfg.training_method = val;
			else if (key == "log_file") cfg.log_file = val;
			else if (key == "force_overwrite") cfg.force_overwrite = str_to_bool(val, false);
			else if (key == "face_detection_method") cfg.face_detection_method = val;
			else if (key == "ignore_failure") cfg.ignore_failure = str_to_bool(val, false);
		} catch (...) {}
	}
	return true;
}

// ==========================================================
// Logging
// ==========================================================

void log_tool(const FacialAuthConfig &cfg, const char* level, const char* fmt, ...) {
	if (!cfg.debug && std::string(level) == "DEBUG") return;

	char buf[1024];
	va_list ap;
	va_start(ap, fmt);
	vsnprintf(buf, sizeof(buf), fmt, ap);
	va_end(ap);

	std::string msg = std::string("[") + level + "] " + buf + "\n";
	std::fwrite(msg.c_str(), 1, msg.size(), stderr);

	if (!cfg.log_file.empty()) {
		std::ofstream out(cfg.log_file, std::ios::app);
		if (out.is_open()) out << msg;
	}
}

void ensure_dirs(const std::string &path) {
	try { fs::create_directories(path); } catch (...) {}
}

// ==========================================================
// Camera & Paths
// ==========================================================

bool open_camera(const FacialAuthConfig &cfg, cv::VideoCapture &cap, std::string &device_used) {
	device_used = cfg.device;
	cap.open(cfg.device);
	if (!cap.isOpened() && cfg.fallback_device) {
		cap.open("/dev/video1");
		if (cap.isOpened()) device_used = "/dev/video1";
	}
	if (!cap.isOpened()) return false;

	cap.set(cv::CAP_PROP_FRAME_WIDTH, cfg.width);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.height);
	return true;
}

std::string fa_user_image_dir(const FacialAuthConfig &cfg, const std::string &user) {
	return join_path(join_path(cfg.basedir, "images"), user);
}

std::string fa_user_model_path(const FacialAuthConfig &cfg, const std::string &user) {
	return join_path(join_path(cfg.basedir, "models"), user + ".xml");
}

// ==========================================================
// FaceRecWrapper
// ==========================================================

FaceRecWrapper::FaceRecWrapper(const std::string &modelType_) : modelType(modelType_) {
	recognizer = cv::face::LBPHFaceRecognizer::create();
}

bool FaceRecWrapper::Load(const std::string &file) {
	try { recognizer->read(file); return true; }
	catch (...) { return false; }
}

bool FaceRecWrapper::Save(const std::string &file) const {
	try { recognizer->write(file); return true; }
	catch (...) { return false; }
}

bool FaceRecWrapper::Train(const std::vector<cv::Mat> &images,
						   const std::vector<int> &labels) {
	try { recognizer->train(images, labels); return true; }
	catch (...) { return false; }
						   }

						   bool FaceRecWrapper::Predict(const cv::Mat &face,
														int &label, double &conf) const {
															try { recognizer->predict(face, label, conf); return true; }
															catch (...) { return false; }
														}

														bool FaceRecWrapper::DetectFace(const cv::Mat &frame, cv::Rect &roi) {
															if (frame.empty() || faceCascade.empty()) return false;

															cv::Mat gray;
															cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
															cv::equalizeHist(gray, gray);

															std::vector<cv::Rect> faces;
															faceCascade.detectMultiScale(gray, faces, 1.08, 4,
																						 0, cv::Size(80, 80));

															if (faces.empty()) return false;
															roi = faces[0];
															return true;
														}
														// ==========================================================
														// fa_capture_images (con numerazione incrementale + Haar cascade)
														// ==========================================================

														bool fa_capture_images(const std::string &user,
																			   const FacialAuthConfig &cfg,
									 bool force,
									 std::string &log)
														{
															std::string device_used;
															cv::VideoCapture cap;

															if (!open_camera(cfg, cap, device_used)) {
																log_tool(cfg, "ERROR", "Failed to open camera: %s", cfg.device.c_str());
		return false;
															}

	log_tool(cfg, "INFO", "Camera opened on %s", device_used.c_str());

	std::string img_dir = fa_user_image_dir(cfg, user);
	ensure_dirs(img_dir);

	// === NUMERAZIONE INCREMENTALE ===
	int start_idx = 0;

	if (!force && !cfg.force_overwrite) {
		for (auto &entry : fs::directory_iterator(img_dir)) {
			if (!entry.is_regular_file()) continue;

			std::string fn = entry.path().filename().string();

			if (fn.size() == 11 &&
				fn.rfind("img_", 0) == 0 &&
				fn.substr(7) == ".png")
			{
				try {
					int idx = std::stoi(fn.substr(4, 3));
					if (idx > start_idx) start_idx = idx;
				} catch (...) {}
			}
		}

		log_tool(cfg, "DEBUG",
				 "Existing max index for user %s = %d",
		   user.c_str(), start_idx);
	} else {
		log_tool(cfg, "DEBUG",
				 "Force overwrite enabled → starting from 1");
	}

	// === CARICAMENTO HAAR CASCADE ===
	FaceRecWrapper rec;

	if (!rec.faceCascade.load(cfg.haar_cascade_path)) {
		log_tool(cfg, "ERROR",
				 "Cannot load Haar cascade file: %s",
		   cfg.haar_cascade_path.c_str());
		return false;
	}

	log_tool(cfg, "INFO", "Loaded Haar cascade: %s",
			 cfg.haar_cascade_path.c_str());

	cv::Mat frame;
	int captured = 0;

	while (captured < cfg.frames) {
		cap >> frame;
		if (frame.empty()) {
			log_tool(cfg, "ERROR", "Empty frame from camera");
			continue;
		}

		cv::Rect face_roi;
		if (!rec.DetectFace(frame, face_roi)) {
			log_tool(cfg, "DEBUG", "No face detected, retrying...");
			sleep_ms(cfg.sleep_ms);
			continue;
		}

		// Ritaglia il volto
		cv::Mat face = frame(face_roi).clone();
		cv::Mat gray;
		cv::cvtColor(face, gray, cv::COLOR_BGR2GRAY);
		cv::equalizeHist(gray, gray);
		// ==========================================================
		// Train user
		// ==========================================================

		bool fa_train_user(const std::string &user,
						   const FacialAuthConfig &cfg,
					 const std::string &method,
					 const std::string &inputDir,
					 const std::string &outputModel,
					 bool force,
					 std::string &log)
		{
			std::string dir = inputDir.empty()
			? fa_user_image_dir(cfg, user)
			: inputDir;

			if (!fs::exists(dir)) {
				log_tool(cfg, "ERROR", "Training directory %s does not exist", dir.c_str());
		return false;
			}

	std::vector<cv::Mat> imgs;
	std::vector<int> labels;

	for (auto &entry : fs::directory_iterator(dir)) {
		if (!entry.is_regular_file()) continue;

		cv::Mat img = cv::imread(entry.path().string(),
								 cv::IMREAD_GRAYSCALE);

		if (!img.empty()) {
			imgs.push_back(img);
			labels.push_back(0); // single user
		}
	}

	if (imgs.empty()) {
		log_tool(cfg, "ERROR", "No images found in %s", dir.c_str());
		return false;
	}

	std::string model = outputModel.empty()
		? fa_user_model_path(cfg, user)
		: outputModel;

	if (file_exists(model) && !force && !cfg.force_overwrite) {
		log_tool(cfg, "ERROR", "Model %s exists (use -f)", model.c_str());
		return false;
	}

	FaceRecWrapper rec(method.empty()
						   ? cfg.training_method
						   : method);

	if (!rec.Train(imgs, labels)) {
		log_tool(cfg, "ERROR", "Training failed");
		return false;
	}

	if (!rec.Save(model)) {
		log_tool(cfg, "ERROR", "Failed to save model %s", model.c_str());
		return false;
	}

	log_tool(cfg, "INFO", "Training complete, saved to %s", model.c_str());
	return true;
		}

// ==========================================================
// Test user
// ==========================================================

bool fa_test_user(const std::string &user,
				  const FacialAuthConfig &cfg,
				  const std::string &modelPath,
				  double &best_conf,
				  int &best_label,
				  std::string &log)
{
	std::string model = modelPath.empty()
		? fa_user_model_path(cfg, user)
		: modelPath;

	if (!file_exists(model)) {
		log_tool(cfg, "ERROR", "Model file missing: %s", model.c_str());
		return false;
	}

	FaceRecWrapper rec;
	if (!rec.Load(model)) {
		log_tool(cfg, "ERROR", "Failed to load model");
		return false;
	}

	if (!rec.faceCascade.load(cfg.haar_cascade_path)) {
		log_tool(cfg, "ERROR", "Cannot load Haar cascade: %s", cfg.haar_cascade_path.c_str());
		return false;
	}

	cv::VideoCapture cap;
	std::string device_used;

	if (!open_camera(cfg, cap, device_used)) {
		log_tool(cfg, "ERROR", "Cannot open camera");
		return false;
	}

	best_conf = 1e9;
	best_label = -1;

	cv::Mat frame;

	for (int i = 0; i < cfg.frames; i++) {
		cap >> frame;
		if (frame.empty()) continue;

		cv::Rect roi;
		if (!rec.DetectFace(frame, roi)) continue;

		cv::Mat face = frame(roi).clone();
		cv::Mat gray;
		cv::cvtColor(face, gray, cv::COLOR_BGR2GRAY);
		cv::equalizeHist(gray, gray);

		int lbl;
		double conf;

		if (!rec.Predict(gray, lbl, conf)) continue;

		if (conf < best_conf) {
			best_conf = conf;
			best_label = lbl;
		}

		if (conf <= cfg.threshold) {
			log_tool(cfg, "INFO", "Authentication SUCCESS (conf %.2f ≤ thr %.2f)",
					 conf, cfg.threshold);
			return true;
		}

		sleep_ms(cfg.sleep_ms);
	}

	log_tool(cfg, "WARN",
			 "Authentication FAILED (best_conf %.2f thr %.2f)",
			 best_conf, cfg.threshold);

	return false;
}
