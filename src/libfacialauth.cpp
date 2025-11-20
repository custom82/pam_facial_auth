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
			else if (key == "width" || key == "frame_width") cfg.width = std::max(64, std::stoi(val));
			else if (key == "height" || key == "frame_height") cfg.height = std::max(64, std::stoi(val));
			else if (key == "threshold") cfg.threshold = std::stod(val);
			else if (key == "timeout") cfg.timeout = std::max(1, std::stoi(val));
			else if (key == "nogui" || key == "disable_gui") cfg.nogui = str_to_bool(val, cfg.nogui);
			else if (key == "debug" || key == "verbose") cfg.debug = str_to_bool(val, cfg.debug);
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
		} catch (const std::exception &e) {
			if (logbuf) *logbuf += "Error parsing line: " + line + " (" + e.what() + ")\n";
		}
	}
	return true;
}

// ==========================================================
// Logging utility
// ==========================================================

void log_tool(const FacialAuthConfig &cfg, const char* level, const char* fmt, ...)
{
	std::string lvl(level ? level : "");

	char buf[1024];
	va_list ap;
	va_start(ap, fmt);
	vsnprintf(buf, sizeof(buf), fmt, ap);
	va_end(ap);

	std::string msg = "[" + lvl + "] " + buf + "\n";

	bool is_error = (lvl == "ERROR");

	// stderr → solo se debug attivo OPPURE se è un errore
	if (cfg.debug || is_error) {
		std::fwrite(msg.c_str(), 1, msg.size(), stderr);
	}

	// File di log → sempre se definito
	if (!cfg.log_file.empty()) {
		std::ofstream logf(cfg.log_file, std::ios::app);
		if (logf.is_open()) {
			logf << msg;
		}
	}
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

// ==========================================================
// Validate that a file is an image (.jpg/.jpeg/.png)
// ==========================================================
bool fa_is_valid_image(const std::string &path) {
	std::string lower;

	lower.reserve(path.size());
	for (char c : path)
		lower.push_back(std::tolower(c));

	return (
		lower.ends_with(".jpg") ||
		lower.ends_with(".jpeg") ||
		lower.ends_with(".png")
	);
}



// ==========================================================
// Camera & Paths
// ==========================================================

bool open_camera(const FacialAuthConfig &cfg, cv::VideoCapture &cap, std::string &device_used) {
	device_used = cfg.device;
	cap.open(cfg.device);
	if (!cap.isOpened() && cfg.fallback_device) {
		log_tool(cfg, "WARN", "Primary device %s failed, trying /dev/video1", cfg.device.c_str());
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
		// ensure directory
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
								   std::string cascadePath;
								   const char *envPath = std::getenv("FACIAL_HAAR_PATH");
								   if (envPath) cascadePath = envPath;
								   if (cascadePath.empty())
									   cascadePath = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml";
								   if (!file_exists(cascadePath))
									   cascadePath = "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml";

								   if (!file_exists(cascadePath) || !faceCascade.load(cascadePath)) {
									   std::cerr << "[FA-ERROR] Failed to load Haar cascade file: " << cascadePath << "\n";
									   return false;
								   }
							   	}

							   cv::Mat gray;
							   cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
							   cv::equalizeHist(gray, gray);

							   std::vector<cv::Rect> faces;
							   faceCascade.detectMultiScale(gray, faces, 1.08, 3, 0, cv::Size(60, 60));

							   if (faces.empty()) return false;

							   faceROI = faces[0];
							   return true;
						   }

						   // ==========================================================
						   // High-level API
						   // ==========================================================

						   bool fa_capture_images(const std::string &user,
												  const FacialAuthConfig &cfg,
								bool force,
								std::string &log,
								const std::string &img_format)
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

	// Determine starting index based on existing img_XXX.* files
	int start_index = 0;
	if (!force && !cfg.force_overwrite) {
		int max_idx = 0;
		for (auto &entry : fs::directory_iterator(img_dir)) {
			if (!entry.is_regular_file())
				continue;

			std::string fname = entry.path().filename().string();
			if (fname.rfind("img_", 0) == 0 && fname.size() >= 8) {
				try {
					int idx = std::stoi(fname.substr(4, 3));
					if (idx > max_idx)
						max_idx = idx;
				} catch (...) {
					// ignore parse errors
				}
			}
		}
		start_index = max_idx;
		log_tool(cfg, "DEBUG",
				 "Existing max image index for user %s is %d",
		   user.c_str(), start_index);
	} else {
		log_tool(cfg, "DEBUG",
				 "Force overwrite enabled, starting from index 1 for user %s",
		   user.c_str());
	}

	FaceRecWrapper rec;
	cv::Mat frame;
	int captured = 0;

	log_tool(cfg, "INFO", "Capturing %d frames", cfg.frames);

	std::string fmt = img_format.empty() ? "png" : img_format;
	for (auto &ch : fmt)
		ch = static_cast<char>(::tolower(static_cast<unsigned char>(ch)));

							   while (captured < cfg.frames) {
								   cap >> frame;
								   if (frame.empty())
									   break;

								   cv::Rect roi;
								   if (!rec.DetectFace(frame, roi)) {
									   log_tool(cfg, "DEBUG", "No face detected");
									   continue;
								   }

								   cv::Mat face = frame(roi).clone();
								   cv::Mat gray;
								   cv::cvtColor(face, gray, cv::COLOR_BGR2GRAY);
								   cv::equalizeHist(gray, gray);

								   char namebuf[64];
								   std::snprintf(namebuf, sizeof(namebuf),
												 "img_%03d.%s", start_index + captured + 1, fmt.c_str());
								   std::string path = join_path(img_dir, namebuf);

								   cv::imwrite(path, gray);
								   log_tool(cfg, "INFO", "Saved %s", path.c_str());
								   ++captured;
								   sleep_ms(cfg.sleep_ms);
							   }

							   return captured > 0;
						   }
						   bool fa_train_user(const std::string &user, const FacialAuthConfig &cfg,
											  const std::string &method, const std::string &inputDir,
							const std::string &outputModel, bool force, std::string &log) {
							   std::string train_dir = inputDir.empty() ? fa_user_image_dir(cfg, user) : inputDir;
							   if (!fs::exists(train_dir)) {
								   log_tool(cfg, "ERROR", "Training dir does not exist: %s", train_dir.c_str());
		return false;
							   }

	std::vector<cv::Mat> images;
	std::vector<int> labels;

	for (auto &entry : fs::directory_iterator(train_dir)) {
		if (!entry.is_regular_file()) continue;
		cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
		if (!img.empty()) {
			images.push_back(img);
			labels.push_back(0);
		}
	}

	if (images.empty()) {
		log_tool(cfg, "ERROR", "No training images found in %s", train_dir.c_str());
		return false;
	}

	FaceRecWrapper rec;
	if (!rec.Train(images, labels)) {
		log_tool(cfg, "ERROR", "Training failed");
		return false;
	}

	std::string model_path = outputModel.empty() ? fa_user_model_path(cfg, user) : outputModel;
	if (!rec.Save(model_path)) {
		log_tool(cfg, "ERROR", "Failed to save model to %s", model_path.c_str());
		return false;
	}

	log_tool(cfg, "INFO", "Training completed, model saved to %s", model_path.c_str());
	return true;
							}

bool fa_test_user(const std::string &user, const FacialAuthConfig &cfg,
				  const std::string &modelPath, double &best_conf,
				  int &best_label, std::string &log) {
	std::string model_file = modelPath.empty() ? fa_user_model_path(cfg, user) : modelPath;
	if (!file_exists(model_file)) {
		log_tool(cfg, "ERROR", "Model file missing for user %s: %s", user.c_str(), model_file.c_str());
		return false;
	}

	FaceRecWrapper rec;
	if (!rec.Load(model_file)) {
		log_tool(cfg, "ERROR", "Failed to load model file: %s", model_file.c_str());
		return false;
	}

	std::string device_used;
	cv::VideoCapture cap;
	if (!open_camera(cfg, cap, device_used)) {
		log_tool(cfg, "ERROR", "Failed to open camera: %s", cfg.device.c_str());
		return false;
	}
	log_tool(cfg, "INFO", "Testing user %s on %s", user.c_str(), device_used.c_str());

	cv::Mat frame;
	best_conf = 1e9;
	best_label = -1;

	for (int i = 0; i < cfg.frames; ++i) {
		cap >> frame;
		if (frame.empty()) continue;

		cv::Rect roi;
		if (!rec.DetectFace(frame, roi)) {
			log_tool(cfg, "DEBUG", "No face detected in frame %d", i);
			continue;
		}

		cv::Mat face = frame(roi).clone();
		cv::Mat gray;
		cv::cvtColor(face, gray, cv::COLOR_BGR2GRAY);
		cv::equalizeHist(gray, gray);

		int label = -1;
		double conf = 0.0;
		if (rec.Predict(gray, label, conf)) {
			log_tool(cfg, "INFO", "Frame %d: label=%d conf=%.2f", i, label, conf);
			if (conf < best_conf) {
				best_conf = conf;
				best_label = label;
			}
			if (conf <= cfg.threshold) {
				log_tool(cfg, "INFO", "Facial authentication SUCCESS (conf=%.2f <= thr=%.2f)", conf, cfg.threshold);
				return true;
			}
		}
		sleep_ms(cfg.sleep_ms);
	}

	log_tool(cfg, "WARN", "Facial authentication FAILED (best_conf=%.2f thr=%.2f)", best_conf, cfg.threshold);
	return false;
				  }

				  // ==========================================================
				  // Maintenance helpers (images/models) and root check
				  // ==========================================================

				  bool fa_clean_images(const FacialAuthConfig &cfg, const std::string &user)
				  {
					  std::string imgdir = fa_user_image_dir(cfg, user);
					  if (!fs::exists(imgdir))
						  return true;

					  try {
						  for (auto &entry : fs::directory_iterator(imgdir)) {
							  if (entry.is_regular_file()) {
								  fs::remove(entry.path());
							  }
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
					  std::string imgdir = fa_user_image_dir(cfg, user);

					  if (!fs::exists(imgdir)) {
						  std::cout << "[INFO] No images for user: " << user << "\n";
						  return;
					  }

					  std::cout << "[INFO] Images for user " << user << ":\n";

					  for (auto &entry : fs::directory_iterator(imgdir)) {
						  if (entry.is_regular_file()) {
							  std::cout << "  " << entry.path().filename().string() << "\n";
						  }
					  }
				  }

				  bool fa_check_root(const char *tool_name)
				  {
					  if (geteuid() != 0) {
						  std::cerr << "Error: " << (tool_name ? tool_name : "this program")
						  << " must be run as root.\n";
						  return false;
					  }
					  return true;
				  }
