// =============================================================
// libfacialauth.cpp - OpenCV 4.12 + YUNet + SFace (ONNX via DNN)
// =============================================================

#include "../include/libfacialauth.h"

// ===== OpenCV =====
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect.hpp>   // Haar + FaceDetectorYN
#include <opencv2/videoio.hpp>
#include <opencv2/face.hpp>        // LBPH/Eigen/Fisher
#include <opencv2/dnn.hpp>         // SFace ONNX

// ===== standard =====
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <cctype>
#include <cstring>
#include <cmath>
#include <getopt.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdarg>

namespace fs = std::filesystem;

// ==========================================================
// Config defaults
// ==========================================================

FacialAuthConfig::FacialAuthConfig()
: basedir("/etc/pam_facial_auth"),
device("/dev/video0"),
fallback_device(true),
width(1280),
height(720),
frames(50),
sleep_ms(50),
debug(false),
nogui(false),
detector_profile("auto"),
haar_cascade_path("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"),
yunet_model("/usr/share/opencv4/dnn/models/face_detection_yunet_2023mar.onnx"),
yunet_model_int8("/usr/share/opencv4/dnn/models/face_detection_yunet_2023mar_int8.onnx"),
sface_model("/usr/share/opencv4/dnn/models/face_recognition_sface_2021dec.onnx"),
sface_model_int8("/usr/share/opencv4/dnn/models/face_recognition_sface_2021dec_int8.onnx"),
sface_prefer_fp32(true),
sface_threshold(0.40),
lbph_threshold(80.0),
eigen_threshold(350.0),
fisher_threshold(150.0),
recognizer("auto"),
log_file(""),
force_overwrite(false)
{}

// ==========================================================
// Utility helpers
// ==========================================================

static std::string trim(const std::string &s) {
	size_t b = s.find_first_not_of(" \t\r\n");
	if (b == std::string::npos) return "";
	size_t e = s.find_last_not_of(" \t\r\n");
	return s.substr(b, e - b + 1);
}

static bool str_to_bool(const std::string &s, bool defval) {
	std::string t = trim(s);
	for (char &c : t)
		c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

	if (t == "1" || t == "true"  || t == "yes" || t == "on")  return true;
	if (t == "0" || t == "false" || t == "no"  || t == "off") return false;
	return defval;
}

static std::string join_path(const std::string &a,
							 const std::string &b)
{
	if (a.empty()) return b;
	if (b.empty()) return a;
	if (a.back() == '/') return a + b;
	return a + "/" + b;
}

static bool file_exists(const std::string &p) {
	struct stat st{};
	return (::stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode));
}

static void ensure_dirs(const std::string &path) {
	if (path.empty()) return;
	try {
		fs::create_directories(path);
	} catch (...) {}
}

static void sleep_ms(int ms) {
	if (ms > 0) usleep(static_cast<useconds_t>(ms) * 1000);
}

// suffix helper (niente std::string::ends_with per evitare C++20)
static bool has_suffix(const std::string &s, const std::string &suf) {
	if (s.size() < suf.size()) return false;
	return std::equal(suf.rbegin(), suf.rend(), s.rbegin());
}

// ==========================================================
// Logging
// ==========================================================

static void log_tool(const FacialAuthConfig &cfg,
					 const char *level,
					 const char *fmt, ...)
{
	char buf[1024];

	va_list ap;
	va_start(ap, fmt);
	vsnprintf(buf, sizeof(buf), fmt, ap);
	va_end(ap);

	std::string lev = level ? level : "";
	std::string msg = "[" + lev + "] " + buf + "\n";

	bool is_err = (lev == std::string("ERROR"));

	if (cfg.debug || is_err)
		std::fwrite(msg.c_str(), 1, msg.size(), stderr);

	if (!cfg.log_file.empty()) {
		std::ofstream logf(cfg.log_file, std::ios::app);
		if (logf.is_open())
			logf << msg;
	}
}

// ==========================================================
// Config reader
// ==========================================================

bool read_kv_config(const std::string &path,
					FacialAuthConfig  &cfg,
					std::string       *logbuf)
{
	std::ifstream in(path);
	if (!in.is_open()) {
		if (logbuf) *logbuf += "Cannot open config: " + path + "\n";
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
			continue;
		}

		try {
			// Normalizza a lowercase
			std::string key_l = key;
			for (char &c : key_l)
				c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

			if (key_l == "basedir")               cfg.basedir = val;
			else if (key_l == "models_dir") {
				fs::path p(val);
				if (p.has_parent_path())
					cfg.basedir = p.parent_path().string();
			}
			else if (key_l == "work_dir")         cfg.basedir = val;

			else if (key_l == "device")           cfg.device = val;
			else if (key_l == "width")            cfg.width  = std::stoi(val);
			else if (key_l == "height")           cfg.height = std::stoi(val);
			else if (key_l == "capture_count"
				|| key_l == "frames")           cfg.frames = std::stoi(val);
			else if (key_l == "capture_delay"
				|| key_l == "sleep_ms")         cfg.sleep_ms = std::stoi(val);
			else if (key_l == "debug")            cfg.debug = str_to_bool(val, cfg.debug);
			else if (key_l == "nogui")            cfg.nogui = str_to_bool(val, cfg.nogui);
			else if (key_l == "fallback_device")  cfg.fallback_device = str_to_bool(val, cfg.fallback_device);

			// detector
			else if (key_l == "detector_profile") cfg.detector_profile = val;
			else if (key_l == "haar_model"
				|| key_l == "haar_cascade_path")
				cfg.haar_cascade_path = val;
			else if (key_l == "yunet_model")      cfg.yunet_model = val;
			else if (key_l == "yunet_model_int8") cfg.yunet_model_int8 = val;

			// SFace
			else if (key_l == "sface_model")      cfg.sface_model = val;
			else if (key_l == "sface_model_int8") cfg.sface_model_int8 = val;
			else if (key_l == "sface_threshold")  cfg.sface_threshold = std::stod(val);
			else if (key_l == "sface_prefer_fp32")cfg.sface_prefer_fp32 = str_to_bool(val, cfg.sface_prefer_fp32);

			// thresholds classici
			else if (key_l == "lbph_threshold")   cfg.lbph_threshold = std::stod(val);
			else if (key_l == "eigen_threshold")  cfg.eigen_threshold = std::stod(val);
			else if (key_l == "fisher_threshold") cfg.fisher_threshold = std::stod(val);

			// recognizer
			else if (key_l == "recognizer")       cfg.recognizer = val;

			// log
			else if (key_l == "log_file")         cfg.log_file = val;
			else if (key_l == "force_overwrite")  cfg.force_overwrite = str_to_bool(val, cfg.force_overwrite);

		} catch (...) {
			if (logbuf) *logbuf += "Bad config line: " + line + "\n";
		}
	}

	// normalizza alcuni campi
	for (char &c : cfg.detector_profile)
		c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
	for (char &c : cfg.recognizer)
		c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

	if (cfg.detector_profile.empty())
		cfg.detector_profile = "auto";
	if (cfg.recognizer.empty())
		cfg.recognizer = "auto";

	return true;
}

// ==========================================================
// Path helpers
// ==========================================================

std::string fa_user_image_dir(const FacialAuthConfig &cfg,
							  const std::string      &user)
{
	return join_path(join_path(cfg.basedir, "images"), user);
}

std::string fa_user_model_path(const FacialAuthConfig &cfg,
							   const std::string      &user)
{
	return join_path(join_path(cfg.basedir, "models"), user + ".xml");
}

// ==========================================================
// Model type detection from XML
// ==========================================================

static std::string fa_detect_model_type(const std::string &xmlPath)
{
	std::ifstream in(xmlPath);
	if (!in.is_open())
		return "lbph";

	std::string line;
	while (std::getline(in, line)) {
		if (line.find("opencv_lbphfaces")   != std::string::npos) return "lbph";
		if (line.find("opencv_eigenfaces")  != std::string::npos) return "eigen";
		if (line.find("opencv_fisherfaces") != std::string::npos) return "fisher";
		if (line.find("opencv_sface_model") != std::string::npos) return "sface";
	}
	return "lbph";
}

// ==========================================================
// FaceRecWrapper IMPLEMENTATION
// ==========================================================

FaceRecWrapper::FaceRecWrapper(const std::string &type)
: modelType(type)
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
			recognizer = cv::face::LBPHFaceRecognizer::create();
			modelType  = "lbph";
		}
		return !recognizer.empty();
	} catch (...) {
		recognizer.release();
		return false;
	}
}

bool FaceRecWrapper::InitCascade(const std::string &cascadePath)
{
	if (cascadePath.empty() || !file_exists(cascadePath))
		return false;

	try {
		return faceCascade.load(cascadePath);
	} catch (...) {
		return false;
	}
}

bool FaceRecWrapper::Load(const std::string &file)
{
	try {
		std::string detected = fa_detect_model_type(file);

		if (detected != modelType) {
			modelType = detected;
			if (!CreateRecognizer())
				return false;
		} else if (recognizer.empty()) {
			if (!CreateRecognizer())
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

	try {
		if (recognizer.empty() &&
			!const_cast<FaceRecWrapper*>(this)->CreateRecognizer())
			return false;

		recognizer->train(images, labels);
		return true;
	} catch (...) {
		return false;
	}
}

bool FaceRecWrapper::Predict(const cv::Mat &face,
							 int &prediction,
							 double &confidence) const
							 {
								 if (face.empty())
									 return false;

								 try {
									 recognizer->predict(face, prediction, confidence);
									 return true;
								 } catch (...) {
									 return false;
								 }
							 }

							 bool FaceRecWrapper::DetectFace(const cv::Mat &frame,
															 cv::Rect &faceROI)
							 {
								 if (frame.empty() || faceCascade.empty())
									 return false;

								 cv::Mat gray;
								 cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
								 cv::equalizeHist(gray, gray);

								 std::vector<cv::Rect> faces;
								 faceCascade.detectMultiScale(gray, faces,
															  1.08, 3,
									  0, cv::Size(60, 60));

								 if (faces.empty())
									 return false;

								 faceROI = faces[0];
								 return true;
							 }

							 // ==========================================================
							 // Camera helper
							 // ==========================================================

							 static bool open_camera(const FacialAuthConfig &cfg,
													 cv::VideoCapture &cap,
								std::string &dev_used)
							 {
								 dev_used = cfg.device;
								 cap.open(cfg.device);

								 if (!cap.isOpened() && cfg.fallback_device) {
									 cap.open("/dev/video1");
									 if (cap.isOpened())
										 dev_used = "/dev/video1";
								 }

								 if (!cap.isOpened())
									 return false;

								 cap.set(cv::CAP_PROP_FRAME_WIDTH,  cfg.width);
								 cap.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.height);
								 return true;
							 }

							 // ==========================================================
							 // L2 helper per SFace / embedding
							 // ==========================================================

							 static cv::Mat fa_l2_norm(const cv::Mat &m)
							 {
								 if (m.empty()) return cv::Mat();

								 cv::Mat flat = m.reshape(1, 1);
								 cv::Mat v32;
								 flat.convertTo(v32, CV_32F);

								 double n = cv::norm(v32, cv::NORM_L2);
								 if (n <= 1e-12) return v32;
								 return v32 / static_cast<float>(n);
							 }

							 static double fa_cosine(const cv::Mat &a, const cv::Mat &b)
							 {
								 if (a.empty() || b.empty())
									 return 1.0;

								 cv::Mat aa = fa_l2_norm(a);
								 cv::Mat bb = fa_l2_norm(b);

								 if (aa.cols != bb.cols)
									 return 1.0;

								 const float *pa = aa.ptr<float>(0);
								 const float *pb = bb.ptr<float>(0);
								 int n = aa.cols;

								 double dot = 0.0;
								 for (int i = 0; i < n; ++i)
									 dot += static_cast<double>(pa[i]) * static_cast<double>(pb[i]);

								 double dist = 1.0 - dot;
								 if (dist < 0.0) dist = 0.0;
								 if (dist > 2.0) dist = 2.0;
								 return dist;
							 }

							 // ==========================================================
							 // YUNet helper (sceglie FP32 / INT8 in base a config)
							 // ==========================================================

							 static bool fa_yunet_available(const FacialAuthConfig &cfg)
							 {
								 return (!cfg.yunet_model.empty()     && file_exists(cfg.yunet_model)) ||
								 (!cfg.yunet_model_int8.empty()&& file_exists(cfg.yunet_model_int8));
							 }

							 static std::string fa_choose_yunet_model(const FacialAuthConfig &cfg)
							 {
								 // Per semplicità: se detector_profile è "yunet_int8" forziamo INT8
								 std::string det = cfg.detector_profile;
								 for (char &c : det)
									 c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

								 if (det == "yunet_int8") {
									 if (!cfg.yunet_model_int8.empty() && file_exists(cfg.yunet_model_int8))
										 return cfg.yunet_model_int8;
								 }

								 // default: preferiamo FP32 se esiste
								 if (!cfg.yunet_model.empty() && file_exists(cfg.yunet_model))
									 return cfg.yunet_model;
								 if (!cfg.yunet_model_int8.empty() && file_exists(cfg.yunet_model_int8))
									 return cfg.yunet_model_int8;

								 return std::string();
							 }

							 // ==========================================================
							 // SFace DNN helper
							 // ==========================================================

							 static std::string fa_choose_sface_model(const FacialAuthConfig &cfg)
							 {
								 if (cfg.sface_prefer_fp32) {
									 if (!cfg.sface_model.empty() && file_exists(cfg.sface_model))
										 return cfg.sface_model;
									 if (!cfg.sface_model_int8.empty() && file_exists(cfg.sface_model_int8))
										 return cfg.sface_model_int8;
								 } else {
									 if (!cfg.sface_model_int8.empty() && file_exists(cfg.sface_model_int8))
										 return cfg.sface_model_int8;
									 if (!cfg.sface_model.empty() && file_exists(cfg.sface_model))
										 return cfg.sface_model;
								 }
								 return std::string();
							 }

							 // ==========================================================
							 // CAPTURE IMAGES (auto / HAAR / YUNET)
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

	// detector_profile: auto / haar / yunet / yunet_int8
	std::string det = cfg.detector_profile;
	for (char &c : det)
		c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

	bool want_yunet = false;
	bool want_haar  = false;

	if (det == "auto" || det.empty()) {
		if (fa_yunet_available(cfg))
			want_yunet = true;
		else
			want_haar = true;
	} else if (det == "yunet" || det == "yunet_int8") {
		want_yunet = true;
	} else if (det == "haar") {
		want_haar = true;
	}

	// --- HAAR fallback / default ---
	FaceRecWrapper haar_rec("lbph");
	bool use_haar = false;

	if (want_haar) {
		if (!cfg.haar_cascade_path.empty() &&
			file_exists(cfg.haar_cascade_path) &&
			haar_rec.InitCascade(cfg.haar_cascade_path))
		{
			use_haar = true;
		} else {
			log_tool(cfg, "ERROR",
					 "HAAR selected but haar_cascade_path is invalid");
			return false;
		}
	}

	// --- YUNet (cv::FaceDetectorYN) ---
	cv::Ptr<cv::FaceDetectorYN> yunet;
	bool yunet_ok = false;
	std::string yunet_path;

	if (want_yunet) {
		yunet_path = fa_choose_yunet_model(cfg);
		if (!yunet_path.empty())
			yunet_ok = true;
		else
			log_tool(cfg, "WARN",
					 "detector_profile=%s but no YUNet model found, fallback to HAAR",
			cfg.detector_profile.c_str());
	}

	if (!use_haar && !yunet_ok) {
		log_tool(cfg, "ERROR", "No valid face detector (YUNet/HAAR) available");
		return false;
	}

	// --- output dir ---
	std::string img_dir = fa_user_image_dir(cfg, user);
	ensure_dirs(img_dir);

	int start_idx = 0;
	if (!force && !cfg.force_overwrite) {
		for (auto &e : fs::directory_iterator(img_dir)) {
			if (!e.is_regular_file()) continue;
			std::string name = e.path().filename().string();
			if (name.size() >= 8 && name.rfind("img_", 0) == 0) {
				try {
					int idx = std::stoi(name.substr(4, 3));
					if (idx > start_idx) start_idx = idx;
				} catch (...) {}
			}
		}
	}

	std::string fmt = img_format.empty() ? "jpg" : img_format;
	for (char &c : fmt)
		c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

	int captured = 0;
	cv::Mat frame;

	while (captured < cfg.frames) {
		cap >> frame;
		if (frame.empty())
			break;

		cv::Rect face_roi;
		bool detected = false;

		// --- YUNet (cv::FaceDetectorYN) ---
		if (yunet_ok) {
			try {
				if (yunet.empty()) {
					yunet = cv::FaceDetectorYN::create(
						yunet_path,
						"",
						frame.size(),
													   0.9f,   // score threshold
										0.3f,   // nms threshold
										5000    // top_k
					);
				}

				yunet->setInputSize(frame.size());
				cv::Mat faces;
				yunet->detect(frame, faces);

				if (!faces.empty() && faces.rows > 0) {
					const float *row = faces.ptr<float>(0);
					float x = row[0], y = row[1], w = row[2], h = row[3];

					face_roi = cv::Rect(
						cv::Point(cvRound(x), cvRound(y)),
										cv::Size (cvRound(w), cvRound(h))
					);
					face_roi &= cv::Rect(0, 0, frame.cols, frame.rows);
					detected = (face_roi.width > 0 && face_roi.height > 0);
				}
			} catch (...) {
				log_tool(cfg, "WARN",
						 "YUNet detection failed, disabling and using HAAR");
				yunet_ok = false;
			}
		}

		// --- HAAR fallback ---
		if (!detected && use_haar) {
			if (haar_rec.DetectFace(frame, face_roi))
				detected = true;
		}

		if (!detected) {
			log_tool(cfg, "DEBUG", "No face detected");
			continue;
		}

		cv::Mat face = frame(face_roi).clone();
		if (face.empty()) continue;

		// grayscale + normalize 200x200 per i metodi classici
		cv::Mat gray;
		cv::cvtColor(face, gray, cv::COLOR_BGR2GRAY);
		cv::equalizeHist(gray, gray);
		if (gray.cols < 60 || gray.rows < 60)
			continue;
		cv::resize(gray, gray, cv::Size(200, 200));

		char namebuf[64];
		std::snprintf(namebuf, sizeof(namebuf),
					  "img_%03d.%s", start_idx + captured + 1, fmt.c_str());
		std::string out_path = join_path(img_dir, namebuf);

		cv::imwrite(out_path, gray);
		log_tool(cfg, "INFO", "Saved %s", out_path.c_str());

		captured++;
		sleep_ms(cfg.sleep_ms);
	}

	return captured > 0;
							 }

							 // ==========================================================
							 // SFace embedding (ONNX via DNN) — OpenCV 4.12
							 // ==========================================================

							 static bool fa_compute_sface_embedding(
								 const FacialAuthConfig &cfg,
								 const std::vector<cv::Mat> &images_bgr,
								 cv::Mat &out_emb,
								 std::string &logbuf)
							 {
								 out_emb.release();

								 std::string modelPath = fa_choose_sface_model(cfg);
								 if (modelPath.empty()) {
									 logbuf += "SFace model missing\n";
									 return false;
								 }

								 if (images_bgr.empty()) {
									 logbuf += "No images for SFace embedding\n";
									 return false;
								 }

								 cv::dnn::Net net;
								 try {
									 net = cv::dnn::readNet(modelPath);
								 } catch (const std::exception &e) {
									 logbuf += std::string("SFace readNet failed: ") + e.what() + "\n";
									 return false;
								 } catch (...) {
									 logbuf += "SFace readNet failed (unknown error)\n";
									 return false;
								 }

								 cv::Mat accum;
								 int count = 0;

								 for (const auto &img0 : images_bgr) {
									 if (img0.empty()) continue;

									 cv::Mat img = img0;
									 if (img.channels() == 1)
										 cv::cvtColor(img0, img, cv::COLOR_GRAY2BGR);

									 cv::Mat resized;
									 cv::resize(img, resized, cv::Size(112, 112));

									 cv::Mat blob = cv::dnn::blobFromImage(
										 resized,
										 1.0 / 255.0,
										 cv::Size(112, 112),
																		   cv::Scalar(0, 0, 0),
																		   true,   // swapRB
												false   // crop
									 );

									 cv::Mat feat;
									 try {
										 net.setInput(blob);
										 feat = net.forward();
									 } catch (...) {
										 continue;
									 }

									 if (feat.empty())
										 continue;

									 feat = feat.reshape(1, 1);
									 feat = fa_l2_norm(feat);

									 if (accum.empty())
										 accum = feat.clone();
									 else
										 accum += feat;

									 count++;
								 }

								 if (accum.empty() || count == 0) {
									 logbuf += "Failed to compute SFace features\n";
									 return false;
								 }

								 accum /= static_cast<float>(count);
								 out_emb = fa_l2_norm(accum);

								 return true;
							 }

							 // ==========================================================
							 // Salvataggio embedding in XML
							 // ==========================================================

							 static bool fa_save_embedding_xml(const std::string &xmlPath,
															   const cv::Mat &emb)
							 {
								 if (emb.empty()) return false;

								 try {
									 ensure_dirs(fs::path(xmlPath).parent_path().string());

									 // Sovrascriviamo e scriviamo header minimale + embedding
									 cv::FileStorage fs(xmlPath, cv::FileStorage::WRITE);
									 if (!fs.isOpened()) return false;

									 fs << "opencv_sface_model" << 1;
									 cv::Mat normed = fa_l2_norm(emb).reshape(1,1).clone();
									 fs << "sface_embedding" << normed;

									 return true;
								 } catch (...) {
									 return false;
								 }
							 }

							 static bool fa_load_embedding_xml(const std::string &xmlPath,
															   cv::Mat &emb)
							 {
								 emb.release();

								 try {
									 cv::FileStorage fs(xmlPath, cv::FileStorage::READ);
									 if (!fs.isOpened()) return false;

									 if (!fs["sface_embedding"].isNone()) {
										 fs["sface_embedding"] >> emb;
										 emb = fa_l2_norm(emb);
										 return !emb.empty();
									 }
								 } catch (...) {}

								 return false;
							 }

							 // ==========================================================
							 // fa_train_user() — training LBPH/Eigen/Fisher + SFace
							 // ==========================================================

							 bool fa_train_user(const std::string &user,
												const FacialAuthConfig &cfg,
						   const std::string &method,
						   const std::string &inputDir,
						   const std::string &outputModel,
						   bool /*force*/,
						   std::string &logbuf)
							 {
								 std::string train_dir = inputDir.empty()
								 ? fa_user_image_dir(cfg, user)
								 : inputDir;

								 if (!fs::exists(train_dir)) {
									 log_tool(cfg, "ERROR", "Training directory missing: %s",
											  train_dir.c_str());
									 return false;
								 }

								 // raccogli path immagini
								 std::vector<std::string> img_paths;
								 for (auto &e : fs::directory_iterator(train_dir)) {
									 if (!e.is_regular_file()) continue;

									 std::string p = e.path().string();
									 std::string low = p;
									 for (char &c : low) c = std::tolower(static_cast<unsigned char>(c));

									 if (has_suffix(low, ".jpg") || has_suffix(low, ".jpeg") || has_suffix(low, ".png"))
										 img_paths.push_back(p);
								 }

								 if (img_paths.empty()) {
									 log_tool(cfg, "ERROR", "No training images found");
									 return false;
								 }

								 std::string m = method;
								 for (char &c : m)
									 c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

								 // ======================================================
								 // SFace training — creare embedding e salvarlo nel XML
								 // ======================================================

								 if (m == "sface") {
									 std::vector<cv::Mat> imgs;

									 for (auto &p : img_paths) {
										 cv::Mat img = cv::imread(p, cv::IMREAD_COLOR);
										 if (img.empty()) {
											 // prova come grayscale
											 img = cv::imread(p, cv::IMREAD_GRAYSCALE);
											 if (!img.empty())
												 cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
										 }
										 if (!img.empty()) imgs.push_back(img);
									 }

									 if (imgs.empty()) {
										 log_tool(cfg, "ERROR", "No valid color images for SFace");
										 return false;
									 }

									 cv::Mat emb;
									 if (!fa_compute_sface_embedding(cfg, imgs, emb, logbuf)) {
										 log_tool(cfg, "ERROR", "Cannot compute SFace embedding");
										 return false;
									 }

									 // percorso output modello
									 std::string out_path = outputModel.empty()
									 ? fa_user_model_path(cfg, user)
									 : outputModel;

									 if (!fa_save_embedding_xml(out_path, emb)) {
										 log_tool(cfg, "ERROR", "Cannot save SFace embedding XML");
										 return false;
									 }

									 log_tool(cfg, "INFO", "SFace model saved to %s", out_path.c_str());
									 return true;
								 }

								 // ======================================================
								 // Metodi Classici LBPH / Eigen / Fisher
								 // ======================================================

								 std::vector<cv::Mat> images;
								 std::vector<int> labels;

								 for (auto &p : img_paths) {
									 cv::Mat img = cv::imread(p, cv::IMREAD_GRAYSCALE);
									 if (img.empty()) continue;

									 cv::equalizeHist(img, img);
									 cv::resize(img, img, cv::Size(200,200));

									 images.push_back(img);
									 labels.push_back(0);
								 }

								 if (images.empty()) {
									 log_tool(cfg, "ERROR", "No valid grayscale images");
									 return false;
								 }

								 std::string type = m;
								 if (type != "eigen" && type != "fisher")
									 type = "lbph";

								 FaceRecWrapper rec(type);
								 if (!rec.CreateRecognizer()) {
									 log_tool(cfg, "ERROR", "Cannot create classic recognizer");
									 return false;
								 }

								 if (!rec.Train(images, labels)) {
									 logbuf += "Training LBPH/Eigen/Fisher failed\n";
									 return false;
								 }

								 std::string out_path = outputModel.empty()
								 ? fa_user_model_path(cfg, user)
								 : outputModel;

								 if (!rec.Save(out_path)) {
									 log_tool(cfg, "ERROR", "Cannot save model file");
									 return false;
								 }

								 log_tool(cfg, "INFO", "Model saved: %s", out_path.c_str());
								 return true;
							 }

							 // ==========================================================
							 // fa_test_user (prima SFace, poi fallback LBPH/Eigen/Fisher)
							 // ==========================================================

							 bool fa_test_user(const std::string &user,
											   const FacialAuthConfig &cfg,
						  const std::string &modelPath,
						  double &best_conf,
						  int &best_label,
						  std::string &logbuf,
						  double threshold_override)
							 {
								 best_conf  = 9999.0;
								 best_label = -1;

								 std::string xml_model =
								 modelPath.empty() ? fa_user_model_path(cfg, user) : modelPath;

								 if (!file_exists(xml_model)) {
									 log_tool(cfg, "ERROR", "Model file missing: %s", xml_model.c_str());
		return false;
								 }

	// ======================================================
	// 1) Tentativo SFace: embedding dentro l'XML
	// ======================================================
	cv::Mat emb_ref;
	if (fa_load_embedding_xml(xml_model, emb_ref) && !emb_ref.empty()) {

		std::string modelPathSFace = fa_choose_sface_model(cfg);
		if (modelPathSFace.empty()) {
			log_tool(cfg, "ERROR",
					 "SFace embedding found in XML, but sface_model is missing");
			// fallback classico dopo
		} else {
			cv::dnn::Net net;
			try {
				net = cv::dnn::readNet(modelPathSFace);
			} catch (const std::exception &e) {
				log_tool(cfg, "ERROR",
						 "Cannot create SFace net: %s", e.what());
			} catch (...) {
				log_tool(cfg, "ERROR",
						 "Cannot create SFace net (unknown error)");
			}

			if (!net.empty()) {
				cv::VideoCapture cap;
				std::string dev_used;

				if (!open_camera(cfg, cap, dev_used)) {
					log_tool(cfg, "ERROR", "Cannot open camera: %s",
							 cfg.device.c_str());
					return false;
				}

				log_tool(cfg, "INFO",
						 "Testing user %s (SFace) on device %s",
						 user.c_str(), dev_used.c_str());

				double thr = (threshold_override > 0.0)
				? threshold_override
				: cfg.sface_threshold;

				if (thr <= 0.0) thr = 0.5;

				// detector_profile per SFace (usiamo la stessa pipeline)
				std::string det = cfg.detector_profile;
				for (char &c : det)
					c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

				bool want_yunet = false;
				bool want_haar  = false;

				if (det == "auto" || det.empty()) {
					if (fa_yunet_available(cfg))
						want_yunet = true;
					else
						want_haar = true;
				} else if (det == "yunet" || det == "yunet_int8") {
					want_yunet = true;
				} else if (det == "haar") {
					want_haar = true;
				}

				cv::Ptr<cv::FaceDetectorYN> yunet;
				bool yunet_ok = false;
				std::string yunet_path;

				if (want_yunet) {
					yunet_path = fa_choose_yunet_model(cfg);
					if (!yunet_path.empty())
						yunet_ok = true;
				}

				// HAAR fallback
				FaceRecWrapper haar("lbph");
				bool use_haar = false;

				if (want_haar) {
					if (!cfg.haar_cascade_path.empty() &&
						file_exists(cfg.haar_cascade_path) &&
						haar.InitCascade(cfg.haar_cascade_path))
					{
						use_haar = true;
					}
				}

				cv::Mat frame;
				for (int i = 0; i < cfg.frames; ++i) {
					cap >> frame;
					if (frame.empty())
						continue;

					cv::Rect roi;
					bool detected = false;

					// --- YUNet ---
					if (yunet_ok) {
						try {
							if (yunet.empty()) {
								yunet = cv::FaceDetectorYN::create(
									yunet_path,
									"",
									frame.size(),
																   0.9f,
										   0.3f,
										   5000
								);
							}

							yunet->setInputSize(frame.size());
							cv::Mat faces;
							yunet->detect(frame, faces);

							if (!faces.empty() && faces.rows > 0) {
								const float *r = faces.ptr<float>(0);
								float x = r[0], y = r[1], w = r[2], h = r[3];

								roi = cv::Rect(
									cv::Point(cvRound(x), cvRound(y)),
											   cv::Size (cvRound(w), cvRound(h))
								);
								roi &= cv::Rect(0, 0, frame.cols, frame.rows);
								detected = (roi.width  > 0 &&
								roi.height > 0);
							}
						} catch (...) {
							log_tool(cfg, "WARN",
									 "YUNet detection failed in test, fallback HAAR");
							yunet_ok = false;
						}
					}

					// --- HAAR fallback ---
					if (!detected && use_haar) {
						if (haar.DetectFace(frame, roi))
							detected = true;
					}

					if (!detected)
						continue;

					cv::Mat face = frame(roi).clone();
					if (face.empty())
						continue;

					cv::Mat resized;
					cv::resize(face, resized, cv::Size(112, 112));

					cv::Mat blob = cv::dnn::blobFromImage(
						resized,
						1.0 / 255.0,
						cv::Size(112, 112),
														  cv::Scalar(0, 0, 0),
														  true,
										   false
					);

					cv::Mat feat;
					try {
						net.setInput(blob);
						feat = net.forward();
					} catch (...) {
						continue;
					}

					if (feat.empty())
						continue;

					feat = feat.reshape(1,1);
					double dist = fa_cosine(emb_ref, feat);
					if (dist < best_conf)
						best_conf = dist;

					best_label = 0;

					if (dist <= thr) {
						log_tool(cfg, "INFO",
								 "Auth success (SFace): dist=%.3f <= %.3f",
								 dist, thr);
						return true;
					}

					sleep_ms(cfg.sleep_ms);
				}

				log_tool(cfg, "WARN",
						 "Auth failed (SFace): best_dist=%.3f thr=%.3f",
						 best_conf, thr);
				// se fallisce, proviamo fallback classico
			}
		}
	}

	// ======================================================
	// 2) Fallback classico: LBPH / Eigen / Fisher
	// ======================================================

	std::string type = fa_detect_model_type(xml_model);

	FaceRecWrapper rec(type);
	if (!rec.CreateRecognizer()) {
		log_tool(cfg, "ERROR", "Cannot create classic recognizer: %s",
				 type.c_str());
		return false;
	}

	if (!rec.Load(xml_model)) {
		log_tool(cfg, "ERROR", "Cannot load model file: %s",
				 xml_model.c_str());
		return false;
	}

	if (cfg.haar_cascade_path.empty() ||
		!file_exists(cfg.haar_cascade_path) ||
		!rec.InitCascade(cfg.haar_cascade_path))
	{
		log_tool(cfg, "ERROR",
				 "HAAR cascade missing or cannot be loaded: %s",
		   cfg.haar_cascade_path.c_str());
		return false;
	}

	cv::VideoCapture cap;
	std::string dev_used;

	if (!open_camera(cfg, cap, dev_used)) {
		log_tool(cfg, "ERROR", "Cannot open camera: %s",
				 cfg.device.c_str());
		return false;
	}

	log_tool(cfg, "INFO",
			 "Testing user %s (classic %s) on %s",
			 user.c_str(), type.c_str(), dev_used.c_str());

	double thr = cfg.lbph_threshold;
	if (type == "eigen")  thr = cfg.eigen_threshold;
	if (type == "fisher") thr = cfg.fisher_threshold;
	if (threshold_override > 0.0)
		thr = threshold_override;

								 cv::Mat frame;
								 for (int i = 0; i < cfg.frames; ++i) {
									 cap >> frame;
									 if (frame.empty())
										 continue;

									 cv::Rect roi;
									 if (!rec.DetectFace(frame, roi))
										 continue;

									 cv::Mat face = frame(roi).clone();
									 if (face.empty())
										 continue;

									 cv::Mat gray;
									 cv::cvtColor(face, gray, cv::COLOR_BGR2GRAY);
									 cv::equalizeHist(gray, gray);
									 cv::resize(gray, gray, cv::Size(200, 200));

									 int    label = -1;
									 double conf  = 9999.0;

									 if (!rec.Predict(gray, label, conf))
										 continue;

									 if (conf < best_conf) {
										 best_conf  = conf;
										 best_label = label;
									 }

									 if (conf <= thr) {
										 log_tool(cfg, "INFO",
												  "Auth success (classic %s): conf=%.3f <= %.3f",
												  type.c_str(), conf, thr);
										 return true;
									 }

									 sleep_ms(cfg.sleep_ms);
								 }

								 log_tool(cfg, "WARN",
										  "Auth failed (classic %s): best_conf=%.3f thr=%.3f",
										  type.c_str(), best_conf, thr);
								 return false;
							 }

							 // ==========================================================
							 // Maintenance helpers
							 // ==========================================================

							 bool fa_clean_images(const FacialAuthConfig &cfg, const std::string &user)
							 {
								 std::string dir = fa_user_image_dir(cfg, user);
								 if (!fs::exists(dir))
									 return true;

								 try {
									 for (auto &e : fs::directory_iterator(dir)) {
										 if (e.is_regular_file())
											 fs::remove(e.path());
									 }
									 return true;
								 } catch (...) {
									 return false;
								 }
							 }

							 bool fa_clean_model(const FacialAuthConfig &cfg, const std::string &user)
							 {
								 std::string p = fa_user_model_path(cfg, user);
								 if (!file_exists(p))
									 return true;

								 try {
									 fs::remove(p);
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
								 for (auto &e : fs::directory_iterator(dir)) {
									 if (e.is_regular_file())
										 std::cout << "  " << e.path().filename().string() << "\n";
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

							 // ==========================================================
							 // CLI: facial_capture
							 // ==========================================================

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
								 << "      --detector NAME   Face detector: auto|haar|yunet|yunet_int8\n"
								 << "      --clean           Remove all user images\n"
								 << "      --reset           Remove user model + images\n"
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

								 int  width_opt   = 0;
								 int  height_opt  = 0;
								 int  frames_opt  = 0;
								 int  sleep_opt   = -1;
								 bool force       = false;
								 bool debug_opt   = false;
								 bool nogui_opt   = false;

								 bool opt_clean   = false;
								 bool opt_reset   = false;

								 enum {
									 OPT_FORMAT   = 1000,
									 OPT_HELP     = 1001,
									 OPT_CLEAN    = 1002,
									 OPT_RESET    = 1003,
									 OPT_DETECTOR = 1004
								 };

								 static struct option long_opts[] = {
									 {"user",     required_argument, nullptr, 'u'},
									 {"device",   required_argument, nullptr, 'd'},
									 {"width",    required_argument, nullptr, 'w'},
									 {"height",   required_argument, nullptr, 'h'},
									 {"frames",   required_argument, nullptr, 'n'},
									 {"sleep",    required_argument, nullptr, 's'},
									 {"force",    no_argument,       nullptr, 'f'},
									 {"nogui",    no_argument,       nullptr, 'g'},
									 {"debug",    no_argument,       nullptr, 'v'},
									 {"config",   required_argument, nullptr, 'c'},
									 {"format",   required_argument, nullptr, OPT_FORMAT},
									 {"clean",    no_argument,       nullptr, OPT_CLEAN},
									 {"reset",    no_argument,       nullptr, OPT_RESET},
									 {"detector", required_argument, nullptr, OPT_DETECTOR},
									 {"help",     no_argument,       nullptr, OPT_HELP},
									 {nullptr,    0,                 nullptr, 0}
								 };

								 int opt;
								 int long_index = 0;

								 while ((opt = getopt_long(argc, argv, "u:d:w:h:n:s:fgvc:", long_opts, &long_index)) != -1) {
									 switch (opt) {
										 case 'u': user = optarg ? optarg : ""; break;
										 case 'd': device_opt = optarg ? optarg : ""; break;
										 case 'w': width_opt  = std::atoi(optarg); break;
										 case 'h': height_opt = std::atoi(optarg); break;
										 case 'n': frames_opt = std::atoi(optarg); break;
										 case 's': sleep_opt  = std::atoi(optarg); break;
										 case 'f': force      = true; break;
										 case 'g': nogui_opt  = true; break;
										 case 'v': debug_opt  = true; break;
										 case 'c': if (optarg) config_path = optarg; break;
										 case OPT_FORMAT:   if (optarg) img_format = optarg; break;
										 case OPT_CLEAN:    opt_clean = true; break;
										 case OPT_RESET:    opt_reset = true; break;
										 case OPT_DETECTOR: if (optarg) cfg.detector_profile = optarg; break;
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
								 if (width_opt  > 0) cfg.width  = width_opt;
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

							 // ==========================================================
							 // CLI: facial_training
							 // ==========================================================

							 static void print_facial_training_usage(const char *p)
							 {
								 std::cout <<
								 "Usage: " << p << " -u <user> -m <method> [options]\n"
								 "\n"
								 "Options:\n"
								 "  -u, --user <name>        Username\n"
								 "  -m, --method <type>      lbph | eigen | fisher | sface\n"
								 "  -i, --input <basedir>    Override basedir for images/models\n"
								 "  -o, --output <file>      Output model XML path\n"
								 "  -f, --force              (unused, reserved)\n"
								 "  -v, --verbose            Verbose output\n"
								 "  -h, --help               Show this message\n";
							 }

							 int facial_training_cli_main(int argc, char *argv[])
							 {
								 FacialAuthConfig cfg;
								 std::string config_path = FACIALAUTH_CONFIG_DEFAULT;

								 std::string user;
								 std::string method;
								 std::string input_dir;
								 std::string output_file;
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
										 case 'u': user   = optarg; break;
										 case 'm': method = optarg; break;
										 case 'i': input_dir   = optarg; break;
										 case 'o': output_file = optarg; break;
										 case 'f': force   = true; break;
										 case 'v': verbose = true; break;
										 case 'h':
											 print_facial_training_usage(argv[0]);
											 return 0;
										 default:
											 print_facial_training_usage(argv[0]);
											 return 1;
									 }
								 }

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

								 std::string m = method;
								 for (char &c : m)
									 c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

								 if (m != "lbph" && m != "eigen" &&
									 m != "fisher" && m != "sface") {
									 std::cerr << "ERROR: Invalid method '" << method << "'\n";
								 return 1;
									 }

									 if (!fa_check_root("facial_training"))
										 return 1;

								 if (!input_dir.empty())
									 cfg.basedir = input_dir;

								 if (output_file.empty())
									 output_file = fa_user_model_path(cfg, user);

								 std::string train_dir = fa_user_image_dir(cfg, user);

								 if (!fa_train_user(user, cfg, m, train_dir, output_file,
									 force, logbuf)) {
									 std::cerr << "Training failed\n";
								 if (!logbuf.empty()) std::cerr << logbuf;
								 return 1;
									 }

									 std::cout << "[OK] Model trained: " << output_file << "\n";
									 return 0;
							 }

							 // ==========================================================
							 // CLI: facial_test
							 // ==========================================================

							 static void print_facial_test_usage(const char *p)
							 {
								 std::cout <<
								 "Usage: " << p << " -u <user> -m <model_path> [options]\n"
								 "\n"
								 "Options:\n"
								 "  -u, --user <user>        Utente da verificare (obbligatorio)\n"
								 "  -m, --model <path>       File modello XML (opzionale, default: models/<user>.xml)\n"
								 "  -c, --config <file>      File di configurazione\n"
								 "                           (default: " FACIALAUTH_CONFIG_DEFAULT ")\n"
								 "  -d, --device <device>    Dispositivo webcam (es. /dev/video0)\n"
								 "      --threshold <value>  Soglia (confidenza o distanza, dipende dal modello)\n"
								 "  -v, --verbose            Modalità verbosa\n"
								 "      --nogui              Disabilita la GUI\n"
								 "  -h, --help               Mostra questo messaggio\n";
							 }

							 int facial_test_cli_main(int argc, char *argv[])
							 {
								 FacialAuthConfig cfg;
								 std::string config_path = FACIALAUTH_CONFIG_DEFAULT;

								 std::string user;
								 std::string device_opt;
								 std::string model_path;
								 std::string logbuf;

								 bool debug_opt = false;
								 bool nogui_opt = false;

								 enum {
									 OPT_THRESHOLD = 2000
								 };

								 double threshold_override = -1.0;

								 struct option long_opts[] = {
									 {"user",      required_argument, nullptr, 'u'},
									 {"model",     required_argument, nullptr, 'm'},
									 {"config",    required_argument, nullptr, 'c'},
									 {"device",    required_argument, nullptr, 'd'},
									 {"threshold", required_argument, nullptr, OPT_THRESHOLD},
									 {"verbose",   no_argument,       nullptr, 'v'},
									 {"nogui",     no_argument,       nullptr, 1},
									 {"help",      no_argument,       nullptr, 'h'},
									 {nullptr,0,nullptr,0}
								 };

								 int opt, idx = 0;
								 while ((opt = getopt_long(argc, argv, "u:m:c:d:vh", long_opts, &idx)) != -1) {
									 switch (opt) {
										 case 'u': user       = optarg; break;
										 case 'm': model_path = optarg; break;
										 case 'c': config_path = optarg; break;
										 case 'd': device_opt = optarg; break;
										 case 'v': debug_opt  = true; break;
										 case 'h':
											 print_facial_test_usage(argv[0]);
											 return 0;
										 case OPT_THRESHOLD:
											 if (optarg)
												 threshold_override = std::atof(optarg);
										 break;
										 case 1: // --nogui
											 nogui_opt = true;
											 break;
										 default:
											 print_facial_test_usage(argv[0]);
											 return 1;
									 }
								 }

								 if (!read_kv_config(config_path, cfg, &logbuf)) {
									 std::cerr << "[ERROR] Cannot read config file: " << config_path << "\n";
									 if (!logbuf.empty()) std::cerr << logbuf;
									 return 1;
								 }

								 if (debug_opt) cfg.debug = true;
								 if (nogui_opt) cfg.nogui = true;
								 if (!device_opt.empty()) cfg.device = device_opt;

								 if (user.empty()) {
									 std::cerr << "ERROR: --user required\n";
									 return 1;
								 }

								 if (!fa_check_root("facial_test"))
									 return 1;

								 if (model_path.empty())
									 model_path = fa_user_model_path(cfg, user);

								 double best_conf  = 0.0;
								 int    best_label = -1;

								 bool ok = fa_test_user(user, cfg, model_path,
														best_conf, best_label,
								logbuf, threshold_override);

								 if (!ok) {
									 std::cerr << "Authentication FAILED (best_conf=" << best_conf << ")\n";
									 if (!logbuf.empty()) std::cerr << logbuf;
									 return 2;
								 }

								 std::cout << "[OK] Authentication SUCCESS (conf=" << best_conf << ")\n";
								 return 0;
							 }

							 // ==========================================================
							 // FINAL UTILITY FUNCTIONS (shared API wrappers)
							 // ==========================================================

							 bool fa_capture(const std::string &user,
											 const FacialAuthConfig &cfg_override,
						std::string &logbuf)
							 {
								 FacialAuthConfig cfg = cfg_override;

								 if (!fa_check_root("fa_capture"))
									 return false;

								 return fa_capture_images(user, cfg, cfg.force_overwrite, logbuf, "jpg");
							 }

							 bool fa_train(const std::string &user,
										   const FacialAuthConfig &cfg_override,
					  const std::string &method,
					  std::string &logbuf)
							 {
								 FacialAuthConfig cfg = cfg_override;

								 if (!fa_check_root("fa_train"))
									 return false;

								 std::string input_dir  = fa_user_image_dir(cfg, user);
								 std::string output_xml = fa_user_model_path(cfg, user);

								 return fa_train_user(user, cfg, method, input_dir,
													  output_xml, cfg.force_overwrite, logbuf);
							 }

							 bool fa_test(const std::string &user,
										  const FacialAuthConfig &cfg_override,
					 double &confidence,
					 std::string &logbuf)
							 {
								 FacialAuthConfig cfg = cfg_override;

								 if (!fa_check_root("fa_test"))
									 return false;

								 std::string model = fa_user_model_path(cfg, user);

								 double best_conf = 0.0;
								 int    best_label = -1;

								 bool ok = fa_test_user(user, cfg, model,
														best_conf, best_label,
								logbuf, -1.0);

								 confidence = best_conf;
								 return ok;
							 }

							 // ==========================================================
							 // END OF FILE
							 // ==========================================================
