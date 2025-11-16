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

bool read_kv_config(const std::string &path, FacialAuthConfig &cfg, std::string *logbuf) {
	std::ifstream in(path);
	if (!in.is_open()) {
		if (logbuf) *logbuf += "Config not found: " + path + "\n";
		return false;
	}
	if (logbuf) *logbuf += "Reading config: " + path + "\n";

	std::string line;
	while (std::getline(in, line)) {
		line = trim(line);
		if (line.empty() || line[0] == '#') continue;
		if (logbuf) *logbuf += "LINE: " + line + "\n";
		std::istringstream iss(line);
		std::string key, val;
		if (!(iss >> key)) continue;
		std::getline(iss, val);
		val = trim(val);

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
			else if (key == "haar_cascade_path") {
				setenv("OPENCV_HAAR_PATH", val.c_str(), 1);
				if (logbuf) *logbuf += "Set HAAR path: " + val + "\n";
			}
		} catch (const std::exception &e) {
			if (logbuf) *logbuf += "Error parsing line: " + line + " (" + e.what() + ")\n";
		}
	}
	return true;
}

void ensure_dirs(const std::string &path) {
	if (path.empty()) return;
	try {
		fs::create_directories(path);
	} catch (...) {}
}

bool file_exists(const std::string &path) {
	struct stat st{};
	return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

std::string join_path(const std::string &a, const std::string &b) {
	if (a.empty()) return b;
	if (b.empty()) return a;
	if (a.back() == '/') return a + b;
	return a + "/" + b;
}

void sleep_ms(int ms) {
	if (ms <= 0) return;
	::usleep(static_cast<useconds_t>(ms) * 1000);
}

void log_tool(bool debug, const char* level, const char* fmt, ...) {
	if (!debug && std::string(level) == "DEBUG") return;
	char buf[1024];
	va_list ap;
	va_start(ap, fmt);
	vsnprintf(buf, sizeof(buf), fmt, ap);
	va_end(ap);
	std::cerr << "[FA-" << level << "] " << buf << std::endl;
}

// camera helper
bool open_camera(const FacialAuthConfig &cfg, cv::VideoCapture &cap, std::string &device_used) {
	device_used = cfg.device;
	cap.open(cfg.device);
	if (!cap.isOpened() && cfg.fallback_device) {
		log_tool(cfg.debug, "WARN", "Primary device %s failed, trying /dev/video1", cfg.device.c_str());
		cap.open("/dev/video1");
		if (cap.isOpened()) device_used = "/dev/video1";
	}
	if (!cap.isOpened()) return false;

	cap.set(cv::CAP_PROP_FRAME_WIDTH, cfg.width);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.height);
	return true;
}

// path helper
std::string fa_user_image_dir(const FacialAuthConfig &cfg, const std::string &user) {
	return join_path(join_path(cfg.basedir, "images"), user);
}

std::string fa_user_model_path(const FacialAuthConfig &cfg, const std::string &user) {
	return join_path(join_path(cfg.basedir, "models"), user + ".xml");
}

// ==========================================================
// FaceRecWrapper
// ==========================================================

FaceRecWrapper::FaceRecWrapper(const std::string &modelType_)
: modelType(modelType_)
{
	recognizer = cv::face::LBPHFaceRecognizer::create();
}

bool FaceRecWrapper::Load(const std::string &modelFile) {
	try {
		recognizer->read(modelFile);
		return true;
	} catch (const std::exception &e) {
		std::cerr << "Error loading model: " << e.what() << std::endl;
		return false;
	}
}

bool FaceRecWrapper::Save(const std::string &modelFile) const {
	try {
		ensure_dirs(fs::path(modelFile).parent_path().string());
		recognizer->write(modelFile);
		return true;
	} catch (const std::exception &e) {
		std::cerr << "Error saving model: " << e.what() << std::endl;
		return false;
	}
}

bool FaceRecWrapper::Train(const std::vector<cv::Mat> &images,
						   const std::vector<int> &labels) {
	if (images.empty() || labels.empty() || images.size() != labels.size())
		return false;
	try {
		recognizer->train(images, labels);
		return true;
	} catch (const std::exception &e) {
		std::cerr << "Error training model: " << e.what() << std::endl;
		return false;
	}
						   }

						   bool FaceRecWrapper::Predict(const cv::Mat &face, int &prediction, double &confidence) const {
							   if (face.empty()) return false;
							   try {
								   recognizer->predict(face, prediction, confidence);
								   return true;
							   } catch (const std::exception &e) {
								   std::cerr << "Error predicting: " << e.what() << std::endl;
								   return false;
							   }
						   }

						   bool FaceRecWrapper::DetectFace(const cv::Mat &frame, cv::Rect &faceROI) {
							   if (frame.empty()) return false;

							   if (faceCascade.empty()) {
								   const char* custom_path = getenv("OPENCV_HAAR_PATH");
								   std::string path = (custom_path && *custom_path) ?
								   std::string(custom_path) :
								   cv::samples::findFile("haarcascade_frontalface_default.xml", false);

								   if (path.empty() || !faceCascade.load(path)) {
									   std::cerr << "Failed to load Haar cascade: " << path << std::endl;
									   return false;
								   }
							   }

							   cv::Mat gray;
							   cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
							   cv::equalizeHist(gray, gray);

							   std::vector<cv::Rect> faces;
							   faceCascade.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(80, 80));

							   if (faces.empty()) return false;
							   faceROI = faces[0];
							   return true;
						   }
