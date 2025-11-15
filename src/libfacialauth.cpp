#include "../include/libfacialauth.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
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

		// Normalize key/value: allow "key value" or "key = value"
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
			else if (key == "model_path") cfg.basedir = val; // optional override
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
	} catch (...) {
		// ignore
	}
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

// In libfacialauth.cpp


bool fa_capture_images(const std::string &user,
					   const FacialAuthConfig &cfg,
					   bool force,
					   std::string &log)
{
	// Open the camera
	cv::VideoCapture cap(cfg.device);
	if (!cap.isOpened()) {
		log += "Failed to open camera: " + cfg.device + "\n";
		return false;
	}

	cv::Mat frame;
	int captured = 0;
	std::string img_dir = fa_user_image_dir(cfg, user);

	// Ensure the directory exists
	ensure_dirs(img_dir);

	// Capture the specified number of frames
	while (captured < cfg.frames) {
		cap >> frame; // Capture a frame
		if (frame.empty()) {
			log += "Empty frame captured.\n";
			break;
		}

		// Save the image to disk
		char filename[64];
		snprintf(filename, sizeof(filename), "img_%03d.png", captured);
		std::string path = join_path(img_dir, filename);

		if (!cv::imwrite(path, frame)) {
			log += "Failed to save image: " + path + "\n";
			return false;
		}

		log += "Captured image: " + path + "\n";
		++captured;
	}

	return captured > 0; // Return true if any images were captured
}


bool fa_train_user(const std::string &user,
				   const FacialAuthConfig &cfg,
				   const std::string &method,
				   const std::string &inputDir,
				   const std::string &outputModel,
				   bool force,
				   std::string &log)
{
	log += "Training user: " + user + "\n";
	log += "Input dir: " + inputDir + "\n";
	log += "Output model: " + outputModel + "\n";

	// Ensure directory exists
	if (!fs::exists(inputDir)) {
		log += "Input directory does not exist: " + inputDir + "\n";
		return false;
	}

	// Collect training images
	std::vector<cv::Mat> images;
	std::vector<int> labels;
	int label_id = 0;

	for (const auto &entry : fs::directory_iterator(inputDir)) {
		if (!entry.is_regular_file()) continue;
		std::string path = entry.path().string();

		cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
		if (img.empty()) {
			log += "Failed to read image: " + path + "\n";
			continue;
		}

		images.push_back(img);
		labels.push_back(label_id);
	}

	if (images.empty()) {
		log += "No valid training images found.\n";
		return false;
	}

	// Create recognizer
	auto recognizer = cv::face::LBPHFaceRecognizer::create();
	recognizer->train(images, labels);

	// Ensure output directory exists
	ensure_dirs(fs::path(outputModel).parent_path().string());

	recognizer->save(outputModel);
	log += "Training completed. Model saved to " + outputModel + "\n";

	return true;
}

bool fa_test_user(const std::string &user,
				  const FacialAuthConfig &cfg,
				  const std::string &modelPath,
				  double &best_conf,
				  int &best_label,
				  std::string &log)
{
	log += "Testing user: " + user + "\n";

	std::string model = modelPath.empty()
	? join_path(join_path(cfg.basedir, "models"), user + ".xml")
	: modelPath;

	if (!file_exists(model)) {
		log += "Model file not found: " + model + "\n";
		return false;
	}

	// Load trained model
	cv::Ptr<cv::face::LBPHFaceRecognizer> recognizer = cv::face::LBPHFaceRecognizer::create();
	try {
		recognizer->read(model);
	} catch (const std::exception &e) {
		log += "Error loading model: " + std::string(e.what()) + "\n";
		return false;
	}

	// Open the webcam
	cv::VideoCapture cap(cfg.device);
	if (!cap.isOpened()) {
		log += "Failed to open camera: " + cfg.device + "\n";
		return false;
	}
	log += "Camera opened successfully.\n";

	cv::CascadeClassifier faceCascade;
	std::string cascadePath = cv::samples::findFile("haarcascade_frontalface_default.xml", false);
	if (cascadePath.empty() || !faceCascade.load(cascadePath)) {
		log += "Failed to load Haar cascade.\n";
		return false;
	}

	best_conf = 1e9;
	best_label = -1;
	bool matched = false;

	auto start = static_cast<int>(::time(nullptr));
	int frames_tried = 0;

	cv::Mat frame;
	while (frames_tried < cfg.frames) {
		cap >> frame;
		if (frame.empty()) {
			log += "Empty frame captured.\n";
			break;
		}

		cv::Mat gray;
		cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
		cv::equalizeHist(gray, gray);

		std::vector<cv::Rect> faces;
		faceCascade.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(80, 80));

		if (!faces.empty()) {
			cv::Rect roi = faces[0];
			cv::Mat face = gray(roi).clone();

			int label = -1;
			double conf = 0.0;
			recognizer->predict(face, label, conf);

			log += "Prediction: label=" + std::to_string(label) +
			", confidence=" + std::to_string(conf) + "\n";

			if (conf < best_conf) {
				best_conf = conf;
				best_label = label;
			}

			if (conf <= cfg.threshold) {
				matched = true;
				if (!cfg.nogui) {
					cv::rectangle(frame, roi, cv::Scalar(0, 255, 0), 2);
					cv::imshow("facial_test", frame);
					cv::waitKey(300);
				}
				break;
			}
		} else if (cfg.debug) {
			log += "No face detected in this frame.\n";
		}

		if (!cfg.nogui) {
			cv::imshow("facial_test", frame);
			if (cv::waitKey(1) == 'q') break;
		}

		++frames_tried;
		int now = static_cast<int>(::time(nullptr));
		if (now - start >= cfg.timeout) {
			log += "Timeout reached.\n";
			break;
		}
	}

	if (!cfg.nogui) cv::destroyAllWindows();
	return matched;
}
