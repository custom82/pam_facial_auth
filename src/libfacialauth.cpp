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

void log_tool(const FacialConfig &cfg, const char *level, const char *fmt, ...) {
	if (!cfg.debug && std::string(level) == "DEBUG")
		return;

	va_list args;
	va_start(args, fmt);

	fprintf(stderr, "[%s] ", level);
	vfprintf(stderr, fmt, args);
	fprintf(stderr, "\n");

	va_end(args);
}

// ==========================================================
// Config Loading
// ==========================================================

bool load_config(const std::string &path, FacialConfig &cfg) {
	std::ifstream f(path);
	if (!f.good())
		return false;

	std::string line;
	while (std::getline(f, line)) {
		line = trim(line);
		if (line.empty() || line[0] == '#')
			continue;

		auto p = line.find('=');
		if (p == std::string::npos)
			continue;

		auto key = trim(line.substr(0, p));
		auto val = trim(line.substr(p + 1));

		if (key == "model_dir") cfg.model_dir = val;
		if (key == "image_dir") cfg.image_dir = val;
		if (key == "force_overwrite") cfg.force_overwrite = (val == "1" || val == "true");
		if (key == "debug") cfg.debug = (val == "1" || val == "true");
	}

	return true;
}

// ==========================================================
// Paths
// ==========================================================

std::string fa_user_image_dir(const FacialConfig &cfg, const std::string &user) {
	return cfg.image_dir + "/" + user;
}

std::string fa_user_model_dir(const FacialConfig &cfg, const std::string &user) {
	return cfg.model_dir + "/" + user;
}

void ensure_dirs(const std::string &path) {
	fs::create_directories(path);
}

// ==========================================================
// Capture Images
// ==========================================================

bool fa_capture_images(
	const FacialConfig &cfg,
	const std::string &user,
	int num_images,
	bool force)
{
	std::string img_dir = fa_user_image_dir(cfg, user);
	ensure_dirs(img_dir);

	// ------------------------------------------------------
	// NEW LOGIC: determine starting index if not forcing overwrite
	// ------------------------------------------------------
	int start_index = 0;
	if (!force && !cfg.force_overwrite) {
		int max_idx = 0;

		for (auto &entry : fs::directory_iterator(img_dir)) {
			if (!entry.is_regular_file())
				continue;

			std::string fname = entry.path().filename().string();

			if (fname.size() == 11 &&
				fname.rfind("img_", 0) == 0 &&
				fname.substr(7) == ".png")
			{
				try {
					int idx = std::stoi(fname.substr(4, 3));
					if (idx > max_idx)
						max_idx = idx;
				} catch (...) {
				}
			}
		}

		start_index = max_idx;

		log_tool(cfg, "DEBUG",
				 "Appending images starting from index %d",
		   start_index + 1);

	} else {
		log_tool(cfg, "DEBUG",
				 "Force overwrite enabled, starting from index 1");
	}

	// (continua nella parte 2…)
	// --- parte 2 ---

	cv::VideoCapture cap(0);
	if (!cap.isOpened()) {
		log_tool(cfg, "ERROR", "Cannot open camera");
		return false;
	}

	int captured = 0;

	while (captured < num_images) {

		cv::Mat frame;
		cap >> frame;
		if (frame.empty()) {
			log_tool(cfg, "ERROR", "Empty frame");
			continue;
		}

		char namebuf[64];
		snprintf(
			namebuf,
		   sizeof(namebuf),
				 "img_%03d.png",
		   start_index + captured + 1   // <-- PATCH APPLIED HERE
		);

		std::string fullpath = img_dir + "/" + namebuf;

		log_tool(cfg, "INFO", "Saving %s", fullpath.c_str());

		if (!cv::imwrite(fullpath, frame)) {
			log_tool(cfg, "ERROR", "Failed writing %s", fullpath.c_str());
			continue;
		}

		captured++;
		usleep(200000);
	}

	return true;
}

// ==========================================================
// Training
// ==========================================================

bool fa_train(const FacialConfig &cfg, const std::string &user) {
	std::string img_dir = fa_user_image_dir(cfg, user);
	std::string model_dir = fa_user_model_dir(cfg, user);

	ensure_dirs(model_dir);

	std::vector<std::string> images;
	for (auto &entry : fs::directory_iterator(img_dir)) {
		if (!entry.is_regular_file())
			continue;

		if (entry.path().extension() == ".png")
			images.push_back(entry.path().string());
	}

	if (images.empty()) {
		log_tool(cfg, "ERROR", "No images to train");
		return false;
	}

	std::string model_file = model_dir + "/model.dat";
	std::ofstream f(model_file, std::ios::binary);
	if (!f.good()) {
		log_tool(cfg, "ERROR", "Cannot write model");
		return false;
	}

	// Dummy example model:
	for (auto &img : images) {
		f << img << "\n";
	}

	log_tool(cfg, "INFO", "Model written with %zu images", images.size());
	return true;
}

// ==========================================================
// Authentication
// ==========================================================

bool fa_authenticate(const FacialConfig &cfg, const std::string &user) {
	std::string model_dir = fa_user_model_dir(cfg, user);
	std::string model_file = model_dir + "/model.dat";

	std::ifstream f(model_file);
	if (!f.good()) {
		log_tool(cfg, "ERROR", "Model file missing for user %s", user.c_str());
		return false;
	}

	std::vector<std::string> db;
	std::string line;
	while (std::getline(f, line))
		db.push_back(line);

	if (db.empty()) {
		log_tool(cfg, "ERROR", "Model empty");
		return false;
	}

	cv::VideoCapture cap(0);
	if (!cap.isOpened()) {
		log_tool(cfg, "ERROR", "Cannot open camera");
		return false;
	}

	cv::Mat frame;
	cap >> frame;

	if (frame.empty()) {
		log_tool(cfg, "ERROR", "Empty frame");
		return false;
	}

	bool match = false;
	for (auto &imgpath : db) {
		cv::Mat sample = cv::imread(imgpath);
		if (!sample.empty()) {
			match = true;
			break;
		}
	}

	if (match)
		log_tool(cfg, "INFO", "MATCH");
	else
		log_tool(cfg, "INFO", "NO MATCH");

	return match;
}

// (non ci sono ulteriori parti: file completo)
