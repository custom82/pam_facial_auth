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
		log_tool(cfg.debug, "WARN", "Primary device %s failed, trying /dev/video1",
				 cfg.device.c_str());
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
	// Per ora solo LBPH
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
								   // usa il classifier di default di OpenCV (installato nel path dei dati)
								   std::string cascadePath = cv::samples::findFile("haarcascade_frontalface_default.xml", false);
								   if (cascadePath.empty() || !faceCascade.load(cascadePath)) {
									   std::cerr << "Failed to load Haar cascade (haarcascade_frontalface_default.xml)\n";
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

						   // ==========================================================
						   // High level API
						   // ==========================================================

						   // helper per numerazione immagini
						   static int fa_find_start_index(const std::string &dir) {
							   int max_idx = 0;
							   if (!fs::exists(dir)) return 1;
							   for (auto &entry : fs::directory_iterator(dir)) {
								   if (!entry.is_regular_file()) continue;
								   auto name = entry.path().filename().string();
								   // formato: img_XXX.png
								   if (name.rfind("img_", 0) == 0) {
									   try {
										   auto pos = name.find('.');
										   std::string num = name.substr(4, pos - 4);
										   int idx = std::stoi(num);
										   if (idx > max_idx) max_idx = idx;
									   } catch (...) {}
								   }
							   }
							   return max_idx + 1;
						   }

						   bool fa_capture_images(const std::string &user,
												  const FacialAuthConfig &cfg,
								bool force,
								std::string &log)
						   {
							   FacialAuthConfig localCfg = cfg; // copia, in caso serva modificarla

							   std::string device_used;
							   cv::VideoCapture cap;
							   if (!open_camera(localCfg, cap, device_used)) {
								   log += "Failed to open camera: " + localCfg.device + "\n";
								   return false;
							   }
							   log += "Camera opened on " + device_used + "\n";

							   std::string img_dir = fa_user_image_dir(localCfg, user);
							   ensure_dirs(img_dir);

							   int idx = force ? 1 : fa_find_start_index(img_dir);

							   FaceRecWrapper rec;
							   cv::Mat frame;
							   int captured = 0;

							   log += "Capturing " + std::to_string(localCfg.frames) + " images to " + img_dir + "\n";

							   while (captured < localCfg.frames) {
								   cap >> frame;
								   if (frame.empty()) {
									   log += "Empty frame from camera\n";
									   break;
								   }

								   cv::Rect roi;
								   if (!rec.DetectFace(frame, roi)) {
									   if (localCfg.debug) log += "No face detected in frame\n";
									   if (!localCfg.nogui) {
										   cv::imshow("facial_capture", frame);
										   if (cv::waitKey(1) == 'q') break;
									   }
									   continue;
								   }

								   cv::Mat face = frame(roi).clone();
								   cv::Mat gray;
								   cv::cvtColor(face, gray, cv::COLOR_BGR2GRAY);
								   cv::equalizeHist(gray, gray);

								   char namebuf[64];
								   snprintf(namebuf, sizeof(namebuf), "img_%03d.png", idx);
								   std::string path = join_path(img_dir, namebuf);
								   if (!cv::imwrite(path, gray)) {
									   log += "Failed to write image: " + path + "\n";
								   } else {
									   log += "Saved: " + path + "\n";
									   ++captured;
									   ++idx;
								   }

								   if (!localCfg.nogui) {
									   cv::rectangle(frame, roi, cv::Scalar(0, 255, 0), 2);
									   cv::imshow("facial_capture", frame);
									   if (cv::waitKey(1) == 'q') break;
								   }

								   sleep_ms(localCfg.sleep_ms);
							   }

							   if (!localCfg.nogui) cv::destroyAllWindows();
							   return captured > 0;
						   }

						   bool fa_train_user(const std::string &user,
											  const FacialAuthConfig &cfg,
							const std::string &method,
							const std::string &inputDir,
							const std::string &outputModel,
							bool force,
							std::string &log)
						   {
							   (void)force; // per ora non controlliamo l'esistenza, sovrascriviamo sempre

							   std::string m = method;
							   for (char &c : m) c = static_cast<char>(::tolower(c));
							   if (m != "lbph") {
								   log += "Only 'lbph' method is supported currently\n";
								   return false;
							   }

							   std::string train_dir = inputDir.empty() ? fa_user_image_dir(cfg, user) : inputDir;
							   if (!fs::exists(train_dir)) {
								   log += "Training dir does not exist: " + train_dir + "\n";
								   return false;
							   }

							   std::vector<cv::Mat> images;
							   std::vector<int> labels;

							   for (auto &entry : fs::directory_iterator(train_dir)) {
								   if (!entry.is_regular_file()) continue;
								   auto path = entry.path().string();
								   cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
								   if (img.empty()) continue;
								   images.push_back(img);
								   labels.push_back(0); // singolo utente
								   if (cfg.debug) log += "Loaded training image: " + path + "\n";
							   }

							   if (images.empty()) {
								   log += "No training images found in " + train_dir + "\n";
								   return false;
							   }

							   FaceRecWrapper rec;
							   if (!rec.Train(images, labels)) {
								   log += "Training failed\n";
								   return false;
							   }

							   std::string model_path = outputModel.empty() ? fa_user_model_path(cfg, user) : outputModel;
							   if (!rec.Save(model_path)) {
								   log += "Failed to save model to " + model_path + "\n";
								   return false;
							   }

							   log += "Model saved to: " + model_path + "\n";
							   return true;
						   }

						   bool fa_test_user(const std::string &user,
											 const FacialAuthConfig &cfg,
						   const std::string &modelPath,
						   double &best_conf,
						   int &best_label,
						   std::string &log)
						   {
							   (void)user; // per ora non usiamo il nome per altro se non logging

							   std::string model = modelPath;
							   if (model.empty()) {
								   model = fa_user_model_path(cfg, user);
							   }

							   if (!file_exists(model)) {
								   log += "Model file does not exist: " + model + "\n";
								   return false;
							   }

							   FaceRecWrapper rec;
							   if (!rec.Load(model)) {
								   log += "Failed to load model: " + model + "\n";
								   return false;
							   }

							   std::string device_used;
							   cv::VideoCapture cap;
							   if (!open_camera(cfg, cap, device_used)) {
								   log += "Failed to open camera: " + cfg.device + "\n";
								   return false;
							   }
							   log += "Camera opened on " + device_used + "\n";

							   cv::Mat frame;
							   cv::Rect roi;

							   best_conf  = 1e9;
							   best_label = -1;
							   bool matched = false;

							   int frames_tried = 0;
							   auto start = static_cast<int>(::time(nullptr));

							   while (true) {
								   cap >> frame;
								   if (frame.empty()) {
									   log += "Empty frame from camera\n";
									   break;
								   }

								   if (!rec.DetectFace(frame, roi)) {
									   if (cfg.debug) log += "No face detected in frame\n";
								   } else {
									   cv::Mat face = frame(roi).clone();
									   cv::Mat gray;
									   cv::cvtColor(face, gray, cv::COLOR_BGR2GRAY);
									   cv::equalizeHist(gray, gray);

									   int label = -1;
									   double conf = 0.0;
									   if (rec.Predict(gray, label, conf)) {
										   log += "Frame#" + std::to_string(frames_tried) +
										   " prediction: label=" + std::to_string(label) +
										   " conf=" + std::to_string(conf) + "\n";
										   if (conf < best_conf) {
											   best_conf  = conf;
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
									   }
								   }

								   if (!cfg.nogui) {
									   cv::imshow("facial_test", frame);
									   if (cv::waitKey(1) == 'q') break;
								   }

								   ++frames_tried;
								   if (frames_tried >= cfg.frames) break;

								   int now = static_cast<int>(::time(nullptr));
								   if (now - start >= cfg.timeout) {
									   log += "Timeout reached\n";
									   break;
								   }
							   }

							   if (!cfg.nogui) cv::destroyAllWindows();
							   return matched;
						   }
