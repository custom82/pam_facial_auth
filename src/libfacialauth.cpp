#include "../include/libfacialauth.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdarg>

namespace fs = std::filesystem;

// ==========================================================
// Helper per logging runtime
// ==========================================================
static inline void append_and_emit(std::string &log,
								   const FacialAuthConfig &cfg,
								   const std::string &line)
{
	log += line;
	log.push_back('\n');
	if (cfg.debug)
		std::cerr << line << std::endl;
}

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

bool read_kv_config(const std::string &path, FacialAuthConfig &cfg, std::string *logbuf) {
	std::ifstream in(path);
	if (!in.is_open()) {
		if (logbuf) logbuf->append("Config not found: " + path + "\n");
		return false;
	}
	if (logbuf) logbuf->append("Reading config: " + path + "\n");

	std::string line;
	while (std::getline(in, line)) {
		line = trim(line);
		if (line.empty() || line[0] == '#') continue;

		size_t pos = line.find('=');
		std::string key, val;
		if (pos != std::string::npos) {
			key = trim(line.substr(0, pos));
			val = trim(line.substr(pos + 1));
		} else {
			std::istringstream iss(line);
			iss >> key;
			std::getline(iss, val);
			val = trim(val);
		}

		try {
			if (key == "basedir") cfg.basedir = val;
			else if (key == "device") cfg.device = val;
			else if (key == "width" || key == "frame_width") cfg.width = std::max(64, std::stoi(val));
			else if (key == "height" || key == "frame_height") cfg.height = std::max(64, std::stoi(val));
			else if (key == "threshold") cfg.threshold = std::stod(val);
			else if (key == "timeout") cfg.timeout = std::max(1, std::stoi(val));
			else if (key == "nogui" || key == "disable_gui") cfg.nogui = str_to_bool(val, cfg.nogui);
			else if (key == "debug" || key == "verbose") cfg.debug = str_to_bool(val, cfg.debug);
			else if (key == "frames") cfg.frames = std::max(1, std::stoi(val));
			else if (key == "fallback_device") cfg.fallback_device = str_to_bool(val, cfg.fallback_device);
			else if (key == "sleep_ms") cfg.sleep_ms = std::max(0, std::stoi(val));
		} catch (const std::exception &e) {
			if (logbuf)
				logbuf->append("Error parsing line: " + line + " (" + e.what() + ")\n");
		}
	}
	return true;
}

void ensure_dirs(const std::string &path) {
	if (path.empty()) return;
	try { fs::create_directories(path); } catch (...) {}
}

bool file_exists(const std::string &path) {
	struct stat st{};
	return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

std::string join_path(const std::string &a, const std::string &b) {
	if (a.empty()) return b;
	if (b.empty()) return a;
	return (a.back() == '/') ? a + b : a + "/" + b;
}

void sleep_ms(int ms) { if (ms > 0) ::usleep(ms * 1000); }

void log_tool(bool debug, const char* level, const char* fmt, ...) {
	char buf[1024];
	va_list ap;
	va_start(ap, fmt);
	vsnprintf(buf, sizeof(buf), fmt, ap);
	va_end(ap);
	if (debug) std::cerr << "[FA-" << level << "] " << buf << std::endl;
}

bool open_camera(const FacialAuthConfig &cfg, cv::VideoCapture &cap, std::string &device_used) {
	device_used = cfg.device;
	cap.open(cfg.device);
	if (!cap.isOpened() && cfg.fallback_device) {
		log_tool(cfg.debug, "WARN", "Primary %s failed, trying /dev/video1", cfg.device.c_str());
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
bool FaceRecWrapper::Load(const std::string &modelFile) {
	try { recognizer->read(modelFile); return true; }
	catch (const std::exception &e) { std::cerr << "Error loading model: " << e.what() << "\n"; return false; }
}
bool FaceRecWrapper::Save(const std::string &modelFile) const {
	try { ensure_dirs(fs::path(modelFile).parent_path().string()); recognizer->write(modelFile); return true; }
	catch (const std::exception &e) { std::cerr << "Error saving model: " << e.what() << "\n"; return false; }
}
bool FaceRecWrapper::Train(const std::vector<cv::Mat> &images, const std::vector<int> &labels) {
	if (images.empty() || images.size() != labels.size()) return false;
	try { recognizer->train(images, labels); return true; }
	catch (const std::exception &e) { std::cerr << "Train error: " << e.what() << "\n"; return false; }
}
bool FaceRecWrapper::Predict(const cv::Mat &face, int &p, double &c) const {
	if (face.empty()) return false;
	try { recognizer->predict(face, p, c); return true; }
	catch (...) { return false; }
}
bool FaceRecWrapper::DetectFace(const cv::Mat &frame, cv::Rect &roi) {
	if (frame.empty()) return false;
	if (faceCascade.empty()) {
		std::string path = cv::samples::findFile("haarcascade_frontalface_default.xml", false);
		if (path.empty() || !faceCascade.load(path)) return false;
	}
	cv::Mat g;
	cv::cvtColor(frame, g, cv::COLOR_BGR2GRAY);
	cv::equalizeHist(g, g);
	std::vector<cv::Rect> faces;
	faceCascade.detectMultiScale(g, faces, 1.1, 3, 0, cv::Size(80, 80));
	if (faces.empty()) return false;
	roi = faces[0];
	return true;
}

// ==========================================================
// High-level API
// ==========================================================
bool fa_capture_images(const std::string &user, const FacialAuthConfig &cfg, bool force, std::string &log) {
	FacialAuthConfig local = cfg;
	std::string dev;
	cv::VideoCapture cap;
	if (!open_camera(local, cap, dev)) {
		append_and_emit(log, local, "Failed to open camera: " + local.device);
		return false;
	}
	append_and_emit(log, local, "Camera opened on " + dev);

	std::string dir = fa_user_image_dir(local, user);
	ensure_dirs(dir);
	int idx = 1;
	FaceRecWrapper rec;
	cv::Mat frame;
	append_and_emit(log, local, "Capturing " + std::to_string(local.frames) + " frames...");

	int captured = 0;
	while (captured < local.frames) {
		cap >> frame;
		if (frame.empty()) break;
		cv::Rect r;
		if (!rec.DetectFace(frame, r)) {
			append_and_emit(log, local, "No face detected");
			continue;
		}
		cv::Mat gray;
		cv::cvtColor(frame(r), gray, cv::COLOR_BGR2GRAY);
		cv::equalizeHist(gray, gray);
		char fn[64];
		snprintf(fn, sizeof(fn), "img_%03d.png", idx++);
		std::string path = join_path(dir, fn);
		if (cv::imwrite(path, gray)) append_and_emit(log, local, "Saved: " + path);
		else append_and_emit(log, local, "Failed to write " + path);
		++captured;
		sleep_ms(local.sleep_ms);
	}
	if (!local.nogui) cv::destroyAllWindows();
	return captured > 0;
}

bool fa_train_user(const std::string &user, const FacialAuthConfig &cfg,
				   const std::string &method, const std::string &inputDir,
				   const std::string &outputModel, bool force, std::string &log)
{
	std::string dir = inputDir.empty() ? fa_user_image_dir(cfg, user) : inputDir;
	if (!fs::exists(dir)) { append_and_emit(log, cfg, "Train dir missing: " + dir); return false; }

	std::vector<cv::Mat> imgs; std::vector<int> labels;
	for (auto &f : fs::directory_iterator(dir)) {
		if (!f.is_regular_file()) continue;
		auto p = f.path().string();
		cv::Mat im = cv::imread(p, cv::IMREAD_GRAYSCALE);
		if (im.empty()) continue;
		imgs.push_back(im); labels.push_back(0);
		if (cfg.debug) append_and_emit(log, cfg, "Loaded: " + p);
	}
	if (imgs.empty()) { append_and_emit(log, cfg, "No images"); return false; }

	FaceRecWrapper rec;
	if (!rec.Train(imgs, labels)) { append_and_emit(log, cfg, "Train failed"); return false; }

	std::string out = outputModel.empty() ? fa_user_model_path(cfg, user) : outputModel;
	if (!rec.Save(out)) { append_and_emit(log, cfg, "Save failed: " + out); return false; }
	append_and_emit(log, cfg, "Model saved: " + out);
	return true;
}

bool fa_test_user(const std::string &user, const FacialAuthConfig &cfg,
				  const std::string &modelPath, double &best_conf, int &best_label,
				  std::string &log)
{
	std::string model = modelPath.empty() ? fa_user_model_path(cfg, user) : modelPath;
	if (!file_exists(model)) { append_and_emit(log, cfg, "Missing model: " + model); return false; }

	FaceRecWrapper rec;
	if (!rec.Load(model)) { append_and_emit(log, cfg, "Cannot load model"); return false; }

	std::string dev;
	cv::VideoCapture cap;
	if (!open_camera(cfg, cap, dev)) { append_and_emit(log, cfg, "Camera open failed"); return false; }

	append_and_emit(log, cfg, "Camera opened on " + dev);
	best_conf = 1e9; best_label = -1;
	bool matched = false;
	int count = 0; time_t start = time(nullptr);
	cv::Mat f; cv::Rect roi;
	while (true) {
		cap >> f;
		if (f.empty()) break;
		if (!rec.DetectFace(f, roi)) { append_and_emit(log, cfg, "No face"); }
		else {
			cv::Mat g; cv::cvtColor(f(roi), g, cv::COLOR_BGR2GRAY);
			cv::equalizeHist(g, g);
			int lbl; double c;
			if (rec.Predict(g, lbl, c)) {
				append_and_emit(log, cfg, "Pred: label=" + std::to_string(lbl) + " conf=" + std::to_string(c));
				if (c < best_conf) { best_conf = c; best_label = lbl; }
				if (c <= cfg.threshold) { matched = true; break; }
			}
		}
		if (++count >= cfg.frames || (time(nullptr) - start) > cfg.timeout) break;
	}
	if (!cfg.nogui) cv::destroyAllWindows();
	return matched;
}
