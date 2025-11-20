#include "../include/libfacialauth.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdarg.h>
#include <sys/stat.h>
#include <thread>
#include <filesystem>

namespace fs = std::filesystem;

// ==========================================================
// Utility
// ==========================================================

std::string trim(const std::string &s) {
	auto a = s.find_first_not_of(" \t\r\n");
	auto b = s.find_last_not_of(" \t\r\n");
	if (a == std::string::npos) return "";
	return s.substr(a, b - a + 1);
}

bool str_to_bool(const std::string &s, bool defval) {
	if (s == "1" || s == "true" || s == "yes" || s == "on") return true;
	if (s == "0" || s == "false" || s == "no" || s == "off") return false;
	return defval;
}

bool file_exists(const std::string &path) {
	return fs::exists(path);
}

std::string join_path(const std::string &a, const std::string &b) {
	return fs::path(a) / b;
}

void ensure_dirs(const std::string &path) {
	fs::create_directories(path);
}

void sleep_ms(int ms) {
	std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

// ==========================================================
// Logging
// ==========================================================

void log_tool(const FacialAuthConfig &cfg, const char *level, const char *fmt, ...) {
	FILE *fp = fopen(cfg.log_file.c_str(), "a");
	if (!fp) return;
	va_list ap;
	va_start(ap, fmt);
	fprintf(fp, "[%s] ", level);
	vfprintf(fp, fmt, ap);
	fprintf(fp, "\n");
	va_end(ap);
	fclose(fp);
}

// ==========================================================
// Config
// ==========================================================

bool read_kv_config(const std::string &path, FacialAuthConfig &cfg, std::string *logbuf) {
	std::ifstream f(path);
	if (!f.is_open()) {
		if (logbuf) *logbuf += "[WARN] Cannot open config file\n";
		return false;
	}

	std::string line;
	while (std::getline(f, line)) {
		line = trim(line);
		if (line.empty() || line[0] == '#') continue;

		auto pos = line.find('=');
		if (pos == std::string::npos) continue;

		std::string key = trim(line.substr(0, pos));
		std::string val = trim(line.substr(pos + 1));

		if (key == "basedir") cfg.basedir = val;
		else if (key == "device") cfg.device = val;
		else if (key == "haar_cascade") cfg.haar_cascade_path = val;
		else if (key == "training_method") cfg.training_method = val;
		else if (key == "threshold") cfg.threshold = std::stod(val);
		else if (key == "frames") cfg.frames = std::stoi(val);
	}
	return true;
}

// ==========================================================
// Paths
// ==========================================================

std::string fa_user_image_dir(const FacialAuthConfig &cfg, const std::string &user) {
	return join_path(cfg.basedir, "images/" + user);
}

std::string fa_user_model_path(const FacialAuthConfig &cfg, const std::string &user) {
	return join_path(cfg.basedir, "models/" + user + ".xml");
}

// ==========================================================
// Camera
// ==========================================================

bool open_camera(const FacialAuthConfig &cfg, cv::VideoCapture &cap, std::string &device_used) {
	cap.open(cfg.device);

	if (cap.isOpened()) {
		device_used = cfg.device;
		return true;
	}

	if (cfg.fallback_device) {
		cap.open(0);
		if (cap.isOpened()) {
			device_used = "/dev/video0";
			return true;
		}
	}

	return false;
}

// ==========================================================
// FaceRecWrapper (unchanged)
// ==========================================================

FaceRecWrapper::FaceRecWrapper(const std::string &modelType_)
: modelType(modelType_)
{
	if (modelType == "lbph") {
		recognizer = cv::face::LBPHFaceRecognizer::create();
	}
	faceCascade.load("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml");
}

bool FaceRecWrapper::Load(const std::string &modelFile) {
	if (!file_exists(modelFile)) return false;
	recognizer->read(modelFile);
	return true;
}

bool FaceRecWrapper::Save(const std::string &modelFile) const {
	ensure_dirs(fs::path(modelFile).parent_path());
	recognizer->write(modelFile);
	return true;
}

bool FaceRecWrapper::Train(const std::vector<cv::Mat> &images, const std::vector<int> &labels) {
	recognizer->train(images, labels);
	return true;
}

bool FaceRecWrapper::Predict(const cv::Mat &face, int &prediction, double &confidence) const {
	recognizer->predict(face, prediction, confidence);
	return true;
}

bool FaceRecWrapper::DetectFace(const cv::Mat &frame, cv::Rect &faceROI) {
	std::vector<cv::Rect> faces;
	faceCascade.detectMultiScale(frame, faces, 1.1, 3);

	if (faces.empty()) return false;
	faceROI = faces[0];
	return true;
}

// ==========================================================
// Capture images (PATCHED)
// ==========================================================

bool fa_capture_images(
	const std::string &user,
	const FacialAuthConfig &cfg,
	bool force,
	std::string &log,
	const std::string &img_format)
{
	std::string imgdir = fa_user_image_dir(cfg, user);
	ensure_dirs(imgdir);

	std::string device_used;
	cv::VideoCapture cap;
	if (!open_camera(cfg, cap, device_used)) {
		log += "[ERROR] Unable to open camera\n";
		return false;
	}

	cap.set(cv::CAP_PROP_FRAME_WIDTH,  cfg.width);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.height);

	log += "[INFO] Camera opened on " + device_used + "\n";

	// ---- Determine existing max index ----
	int max_id = 0;
	for (auto &entry : fs::directory_iterator(imgdir)) {
		if (!entry.is_regular_file()) continue;

		std::string name = entry.path().filename().string();

		// Accept both .jpg and .png
		if (name.rfind("image_", 0) == 0) {
			size_t p1 = name.find('_');
			size_t p2 = name.find('.');
			if (p1 != std::string::npos && p2 != std::string::npos) {
				int id = std::stoi(name.substr(p1 + 1, p2 - (p1 + 1)));
				if (id > max_id) max_id = id;
			}
		}
	}

	log += "[DEBUG] Last image ID = " + std::to_string(max_id) + "\n";

	// ---- Capture loop ----

	for (int i = 1; i <= cfg.frames; i++) {
		cv::Mat frame;
		cap >> frame;

		if (frame.empty()) {
			log += "[WARN] Empty frame, retrying\n";
			i--;
			sleep_ms(cfg.sleep_ms);
			continue;
		}

		char fname[256];
		snprintf(fname, sizeof(fname), "image_%04d.%s", max_id + i, img_format.c_str());

		std::string path = join_path(imgdir, fname);

		if (!cv::imwrite(path, frame)) {
			log += "[ERROR] Cannot write image " + std::string(fname) + "\n";
			return false;
		}

		log += "[INFO] Saved " + std::string(fname) + "\n";

		sleep_ms(cfg.sleep_ms);
	}

	return true;
}


// ==========================================================
// Training
// ==========================================================

bool fa_train_user(const std::string &user,
				   const FacialAuthConfig &cfg,
				   const std::string &method,
				   const std::string &inputDir,
				   const std::string &outputModel,
				   bool force,
				   std::string &log)
{
	std::vector<cv::Mat> images;
	std::vector<int> labels;

	for (auto &entry : fs::directory_iterator(inputDir)) {
		if (!entry.is_regular_file()) continue;

		cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
		if (img.empty()) continue;

		images.push_back(img);
		labels.push_back(0);
	}

	if (images.empty()) {
		log += "[ERROR] No images found\n";
		return false;
	}

	FaceRecWrapper rec(method);

	if (!rec.Train(images, labels)) {
		log += "[ERROR] Training failed\n";
		return false;
	}

	rec.Save(outputModel);
	log += "[INFO] Model saved\n";

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
	FaceRecWrapper rec(cfg.training_method);

	if (!rec.Load(modelPath)) {
		log += "[ERROR] Cannot load model\n";
		return false;
	}

	std::string device_used;
	cv::VideoCapture cap;
	if (!open_camera(cfg, cap, device_used)) {
		log += "[ERROR] Cannot open camera\n";
		return false;
	}

	cap.set(cv::CAP_PROP_FRAME_WIDTH,  cfg.width);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.height);

	cv::Mat frame;
	cap >> frame;

	if (frame.empty()) {
		log += "[ERROR] Empty frame\n";
		return false;
	}

	cv::Rect faceROI;
	if (!rec.DetectFace(frame, faceROI)) {
		log += "[ERROR] No face detected\n";
		return false;
	}

	cv::Mat face = frame(faceROI).clone();
	cv::cvtColor(face, face, cv::COLOR_BGR2GRAY);

	if (!rec.Predict(face, best_label, best_conf)) {
		log += "[ERROR] Prediction failed\n";
		return false;
	}

	char buf[256];
	snprintf(buf, sizeof(buf),
			 "[DEBUG] Predicted label=%d, confidence=%.2f threshold=%.2f",
		  best_label, best_conf, cfg.threshold);

	log += buf;
	log += "\n";

	return (best_conf < cfg.threshold);
}
