#include "../include/libfacialauth.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/face.hpp>              // LBPH, Eigen, Fisher
#include <opencv2/objdetect/face.hpp>    // FaceDetectorYN, FaceRecognizerSF
#include <opencv2/dnn.hpp>               // DNN backend/target (CUDA/CPU)

#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <cstdarg>
#include <cctype>
#include <cstring>
#include <cfloat>
#include <getopt.h>
#include <sys/stat.h>
#include <unistd.h>

namespace fs = std::filesystem;

// ==========================================================
// Utility
// ==========================================================

std::string trim(const std::string &s) {
	size_t b = s.find_first_not_of(" \t\r\n");
	if (b == std::string::npos)
		return "";

	size_t e = s.find_last_not_of(" \t\r\n");
	return s.substr(b, e - b + 1);
}

bool str_to_bool(const std::string &s, bool defval) {
	std::string t = trim(s);
	for (char &c : t)
		c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

	if (t == "1" || t == "true"  || t == "yes" || t == "on")  return true;
	if (t == "0" || t == "false" || t == "no"  || t == "off") return false;
	return defval;
}

std::string join_path(const std::string &a, const std::string &b) {
	if (a.empty()) return b;
	if (b.empty()) return a;
	if (a.back() == '/') return a + b;
	return a + "/" + b;
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

static void sleep_ms(int ms) {
	if (ms > 0)
		usleep(static_cast<useconds_t>(ms) * 1000);
}

// ==========================================================
// Logging utility
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
			if (key == "basedir") {
				cfg.basedir = val;
			}
			else if (key == "device") cfg.device = val;
			else if (key == "width")  cfg.width  = std::max(64, std::stoi(val));
			else if (key == "height") cfg.height = std::max(64, std::stoi(val));
			else if (key == "frames") cfg.frames = std::max(1,  std::stoi(val));
			else if (key == "sleep_ms") cfg.sleep_ms = std::max(0, std::stoi(val));
			else if (key == "debug") cfg.debug = str_to_bool(val, cfg.debug);
			else if (key == "nogui") cfg.nogui = str_to_bool(val, cfg.nogui);
			else if (key == "fallback_device") cfg.fallback_device = str_to_bool(val, cfg.fallback_device);
			else if (key == "model_path") cfg.model_path = val;
			else if (key == "haar_cascade_path") cfg.haar_cascade_path = val;

			// DNN pipeline (YUNet + SFace)
			else if (key == "detector_profile") cfg.detector_profile = val;
			else if (key == "yunet_model")      cfg.yunet_model      = val;
			else if (key == "sface_model")      cfg.sface_model      = val;
			else if (key == "sface_threshold")  cfg.sface_threshold  = std::stod(val);
			else if (key == "embeddings_dir")   cfg.embeddings_dir   = val;

			else if (key == "training_method")  cfg.training_method  = val;
			else if (key == "log_file")         cfg.log_file        = val;
			else if (key == "force_overwrite")  cfg.force_overwrite = str_to_bool(val, cfg.force_overwrite);
			else if (key == "ignore_failure")   cfg.ignore_failure  = str_to_bool(val, cfg.ignore_failure);

			// vecchio parametro "threshold" -> usa LBPH
			else if (key == "threshold")        cfg.lbph_threshold = std::stod(val);
			else if (key == "lbph_threshold")   cfg.lbph_threshold = std::stod(val);
			else if (key == "eigen_threshold")  cfg.eigen_threshold = std::stod(val);
			else if (key == "fisher_threshold") cfg.fisher_threshold = std::stod(val);
			else if (key == "eigen_components")  cfg.eigen_components  = std::stoi(val);
			else if (key == "fisher_components") cfg.fisher_components = std::stoi(val);
		}
		catch (const std::exception &e) {
			if (logbuf)
				*logbuf += "Invalid line: " + line + " (" + e.what() + ")\n";
		}
	}

	return true;
}

// ==========================================================
// Model type detection from XML
// ==========================================================

std::string fa_detect_model_type(const std::string &xmlPath)
{
	std::ifstream in(xmlPath);
	if (!in.is_open())
		return "lbph";

	std::string line;
	while (std::getline(in, line)) {
		if (line.find("opencv_lbphfaces")   != std::string::npos) return "lbph";
		if (line.find("opencv_eigenfaces")  != std::string::npos) return "eigen";
		if (line.find("opencv_fisherfaces") != std::string::npos) return "fisher";
	}
	return "lbph";
}

// ==========================================================
// Path helpers
// ==========================================================

std::string fa_user_image_dir(const FacialAuthConfig &cfg,
							  const std::string &user)
{
	return join_path(join_path(cfg.basedir, "images"), user);
}

std::string fa_user_model_path(const FacialAuthConfig &cfg,
							   const std::string &user)
{
	return join_path(join_path(cfg.basedir, "models"), user + ".xml");
}

std::string fa_user_embedding_path(const FacialAuthConfig &cfg,
								   const std::string &user)
{
	std::string base = cfg.embeddings_dir.empty()
	? join_path(cfg.basedir, "embeddings")
	: cfg.embeddings_dir;
	return join_path(base, user + ".bin");
}

// ==========================================================
// FaceRecWrapper IMPLEMENTATION (LBPH / Eigen / Fisher + HAAR)
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
			// default/fallback LBPH
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
	if (cascadePath.empty())
		return false;
	if (!file_exists(cascadePath))
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
		if (recognizer.empty() && !const_cast<FaceRecWrapper*>(this)->CreateRecognizer())
			return false;

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
								 if (face.empty())
									 return false;

								 try {
									 recognizer->predict(face, pred, conf);
									 return true;
								 } catch (...) {
									 return false;
								 }
							 }

							 bool FaceRecWrapper::DetectFace(const cv::Mat &frame, cv::Rect &faceROI)
							 {
								 if (frame.empty())
									 return false;

								 if (faceCascade.empty())
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
							 // CAPTURE IMAGES (HAAR / YUNET_CPU / YUNET_CUDA)
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

	// Setup detection pipeline in base a cfg.detector_profile
	std::string det = cfg.detector_profile;
	for (char &c : det)
		c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

	bool want_yunet = (det == "yunet_cpu" || det == "yunet_cuda");

	cv::Ptr<cv::FaceDetectorYN> yunet;
	bool yunet_ok = false;

	FaceRecWrapper haar_det("lbph");
	bool use_haar = false;

	// Se vogliamo YUNet, proviamo a usarlo (CPU o CUDA)
	if (want_yunet && !cfg.yunet_model.empty() && file_exists(cfg.yunet_model)) {
		yunet_ok = true; // verrà creato al primo frame (serve frame.size())
	} else if (want_yunet) {
		log_tool(cfg, "WARN",
				 "detector_profile=%s ma yunet_model non valido, fallback a HAAR",
				 cfg.detector_profile.c_str());
		want_yunet = false;
	}

	// HAAR disponibile come fallback / default?
	if (!want_yunet || !yunet_ok) {
		if (!cfg.haar_cascade_path.empty() && file_exists(cfg.haar_cascade_path)) {
			if (haar_det.InitCascade(cfg.haar_cascade_path)) {
				use_haar = true;
			} else {
				log_tool(cfg, "ERROR",
						 "Cannot load HAAR cascade from %s",
			 cfg.haar_cascade_path.c_str());
				if (!want_yunet) {
					return false;
				}
			}
		} else if (!want_yunet) {
			log_tool(cfg, "ERROR",
					 "detector_profile=haar ma haar_cascade_path mancante o invalido");
			return false;
		}
	}

	if (!yunet_ok && !use_haar) {
		log_tool(cfg, "ERROR",
				 "Nessun face detector disponibile (né YUNet né HAAR)");
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
									 if (frame.empty())
										 break;

									 cv::Rect roi;
									 bool detected = false;

									 // 1) YUNet (se abilitato)
									 if (yunet_ok) {
										 try {
											 if (yunet.empty()) {
												 bool want_cuda = (det == "yunet_cuda");

												 int backend_id = cv::dnn::DNN_BACKEND_OPENCV;
												 int target_id  = cv::dnn::DNN_TARGET_CPU;

												 if (want_cuda) {
													 backend_id = cv::dnn::DNN_BACKEND_CUDA;
													 target_id  = cv::dnn::DNN_TARGET_CUDA;
												 }

												 try {
													 yunet = cv::FaceDetectorYN::create(
														 cfg.yunet_model,
												 "",
												 frame.size(),
																						0.9f, 0.3f, 5000,
												 backend_id,
												 target_id
													 );
												 } catch (...) {
													 if (want_cuda) {
														 log_tool(cfg, "WARN",
																  "YUNet CUDA init failed, falling back to CPU");
														 try {
															 yunet = cv::FaceDetectorYN::create(
																 cfg.yunet_model,
												   "",
												   frame.size(),
																								0.9f, 0.3f, 5000,
												   cv::dnn::DNN_BACKEND_OPENCV,
												   cv::dnn::DNN_TARGET_CPU
															 );
														 } catch (...) {
															 yunet.release();
														 }
													 } else {
														 yunet.release();
													 }
												 }
											 }

											 if (!yunet.empty()) {
												 yunet->setInputSize(frame.size());
												 cv::Mat faces;
												 yunet->detect(frame, faces);

												 if (!faces.empty() && faces.rows > 0) {
													 const float *row = faces.ptr<float>(0);
													 float x = row[0];
													 float y = row[1];
													 float w = row[2];
													 float h = row[3];

													 roi = cv::Rect(
														 cv::Point(cvRound(x), cvRound(y)),
																	cv::Size(cvRound(w), cvRound(h))
													 );
													 roi &= cv::Rect(0, 0, frame.cols, frame.rows);
													 if (roi.width > 0 && roi.height > 0)
														 detected = true;
												 }
											 }
										 } catch (...) {
											 log_tool(cfg, "WARN",
													  "YUNet detection failed, fallback HAAR");
											 yunet_ok = false;
										 }
									 }

									 // 2) Fallback HAAR se necessario
									 if (!detected && use_haar) {
										 if (haar_det.DetectFace(frame, roi))
											 detected = true;
									 }

									 if (!detected) {
										 log_tool(cfg, "DEBUG", "No face detected");
										 continue;
									 }

									 cv::Mat face = frame(roi).clone();
									 if (face.empty())
										 continue;

									 cv::Mat gray;
									 cv::cvtColor(face, gray, cv::COLOR_BGR2GRAY);
									 cv::equalizeHist(gray, gray);

									 if (gray.cols < 60 || gray.rows < 60)
										 continue;
									 cv::resize(gray, gray, cv::Size(200, 200));

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
							 // TRAIN MODEL (classico + SFace embedding binario)
							 // ==========================================================

							 // --- funzioni di supporto SFace (embedding) ---

							 static cv::Mat fa_l2_normalize(const cv::Mat &v)
							 {
								 if (v.empty())
									 return cv::Mat();

								 cv::Mat v_flat = v.reshape(1, 1);
								 cv::Mat v32;
								 v_flat.convertTo(v32, CV_32F);

								 double norm = cv::norm(v32, cv::NORM_L2);
								 if (norm <= 1e-12)
									 return v32;

								 return v32 / static_cast<float>(norm);
							 }

							 static double fa_cosine_distance(const cv::Mat &a, const cv::Mat &b)
							 {
								 if (a.empty() || b.empty())
									 return 1.0;

								 cv::Mat aa = fa_l2_normalize(a);
								 cv::Mat bb = fa_l2_normalize(b);

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

							 static bool fa_save_embedding_binary(const std::string &path, const cv::Mat &emb)
							 {
								 if (emb.empty())
									 return false;

								 cv::Mat v = fa_l2_normalize(emb);
								 cv::Mat v32;
								 v.convertTo(v32, CV_32F);
								 v32 = v32.reshape(1, 1);

								 int32_t length = static_cast<int32_t>(v32.cols);

								 try {
									 ensure_dirs(fs::path(path).parent_path().string());
									 std::ofstream out(path, std::ios::binary);
									 if (!out.is_open())
										 return false;

									 out.write(reinterpret_cast<const char*>(&length), sizeof(length));
									 out.write(reinterpret_cast<const char*>(v32.ptr<float>(0)),
											   static_cast<std::streamsize>(length * sizeof(float)));
									 return static_cast<bool>(out);
								 } catch (...) {
									 return false;
								 }
							 }

							 static bool fa_load_embedding_binary(const std::string &path, cv::Mat &emb)
							 {
								 emb.release();
								 try {
									 std::ifstream in(path, std::ios::binary);
									 if (!in.is_open())
										 return false;

									 int32_t length = 0;
									 in.read(reinterpret_cast<char*>(&length), sizeof(length));
									 if (!in || length <= 0 || length > 4096)
										 return false;

									 cv::Mat v(1, length, CV_32F);
									 in.read(reinterpret_cast<char*>(v.ptr<float>(0)),
											 static_cast<std::streamsize>(length * sizeof(float)));

									 if (!in)
										 return false;

									 emb = fa_l2_normalize(v);
									 return true;
								 } catch (...) {
									 return false;
								 }
							 }

							 static bool fa_compute_sface_embedding(const FacialAuthConfig &cfg,
																	const std::vector<cv::Mat> &images_bgr,
											   cv::Mat &out_emb,
											   std::string &logbuf)
							 {
								 out_emb.release();

								 if (cfg.sface_model.empty() || !file_exists(cfg.sface_model)) {
									 if (!cfg.sface_model.empty())
										 logbuf += "SFace model not found: " + cfg.sface_model + "\n";
									 return false;
								 }

								 if (images_bgr.empty()) {
									 logbuf += "No images provided for SFace embedding\n";
									 return false;
								 }

								 cv::Ptr<cv::FaceRecognizerSF> sface;
								 try {
									 // training: usiamo sempre CPU per semplicità
									 sface = cv::FaceRecognizerSF::create(cfg.sface_model, "",
																		  cv::dnn::DNN_BACKEND_OPENCV,
											   cv::dnn::DNN_TARGET_CPU);
								 } catch (const std::exception &e) {
									 logbuf += std::string("Cannot create SFace recognizer: ") + e.what() + "\n";
									 return false;
								 } catch (...) {
									 logbuf += "Cannot create SFace recognizer (unknown error)\n";
									 return false;
								 }

								 cv::Mat accum;
								 int count = 0;

								 for (const auto &img_bgr : images_bgr) {
									 if (img_bgr.empty())
										 continue;

									 cv::Mat resized;
									 cv::resize(img_bgr, resized, cv::Size(112, 112));

									 cv::Mat feat;
									 try {
										 sface->feature(resized, feat);
									 } catch (...) {
										 continue;
									 }

									 if (feat.empty())
										 continue;

									 feat = fa_l2_normalize(feat);

									 if (accum.empty())
										 accum = feat.clone();
									 else
										 accum += feat;

									 ++count;
								 }

								 if (accum.empty() || count == 0) {
									 logbuf += "Failed to compute SFace embedding (no valid features)\n";
									 return false;
								 }

								 accum /= static_cast<float>(count);
								 out_emb = fa_l2_normalize(accum);
								 return true;
							 }
							 // ----------------------------------------------------------

							 bool fa_train_user(const std::string &user,
												const FacialAuthConfig &cfg,
						   const std::string &method,
						   const std::string &inputDir,
						   const std::string &outputModel,
						   bool /*force*/,
						   std::string &logbuf)
							 {
								 // Directory immagini
								 std::string train_dir =
								 inputDir.empty() ? fa_user_image_dir(cfg, user) : inputDir;

								 if (!fs::exists(train_dir)) {
									 log_tool(cfg, "ERROR", "Training directory missing: %s",
											  train_dir.c_str());
									 return false;
								 }

								 std::vector<std::string> img_paths;

								 auto has_suffix = [](const std::string &s, const char *suf){
									 size_t ls = s.size(), lf = strlen(suf);
									 return (ls >= lf && s.compare(ls - lf, lf, suf) == 0);
								 };

								 for (auto &entry : fs::directory_iterator(train_dir)) {
									 if (!entry.is_regular_file()) continue;

									 std::string path = entry.path().string();
									 std::string lower = path;
									 for (char &c: lower) c = (char)tolower((unsigned char)c);

									 if (!(has_suffix(lower, ".jpg") ||
										 has_suffix(lower, ".jpeg") ||
										 has_suffix(lower, ".png")))
										 continue;

									 img_paths.push_back(path);
								 }

								 if (img_paths.empty()) {
									 log_tool(cfg, "ERROR", "No valid training images found");
									 return false;
								 }

								 std::string method_l = method;
								 for (char &c : method_l)
									 c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

								 // ==========================
								 // SFace (DNN embeddings)
								 // ==========================
								 if (method_l == "sface") {
									 std::vector<cv::Mat> images_bgr;

									 for (const auto &path : img_paths) {
										 cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
										 if (img.empty())
											 continue;
										 if (img.cols < 60 || img.rows < 60)
											 continue;
										 images_bgr.push_back(img);
									 }

									 if (images_bgr.empty()) {
										 log_tool(cfg, "ERROR", "No valid images for SFace training");
										 return false;
									 }

									 cv::Mat emb;
									 if (!fa_compute_sface_embedding(cfg, images_bgr, emb, logbuf)) {
										 log_tool(cfg, "ERROR", "Cannot compute SFace embedding");
										 return false;
									 }

									 std::string emb_out =
									 outputModel.empty() ? fa_user_embedding_path(cfg, user) : outputModel;

									 if (!fa_save_embedding_binary(emb_out, emb)) {
										 log_tool(cfg, "ERROR", "Cannot save SFace embedding: %s",
												  emb_out.c_str());
										 return false;
									 }

									 log_tool(cfg, "INFO", "SFace embedding saved to %s", emb_out.c_str());
									 return true;
								 }

								 // ==========================
								 // Metodi classici (LBPH/Eigen/Fisher)
								 // ==========================
								 std::vector<cv::Mat> images;
								 std::vector<int>     labels;

								 for (const auto &path : img_paths) {
									 cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
									 if (img.empty()) continue;

									 if (img.cols < 60 || img.rows < 60)
										 continue;

									 cv::equalizeHist(img, img);
									 cv::resize(img, img, cv::Size(200, 200)); // uniforme

									 if (method_l == "fisher") {
										 images.push_back(img);
										 labels.push_back(0);

										 cv::Mat blur, gamma;
										 cv::GaussianBlur(img, blur, cv::Size(5,5), 0);

										 img.convertTo(gamma, -1, 1.15, 2); // leggero aumento contrasto

										 images.push_back(blur);
										 labels.push_back(1);

										 images.push_back(gamma);
										 labels.push_back(1);

									 } else {
										 // LBPH ed Eigen → mono-classe
										 images.push_back(img);
										 labels.push_back(0);
									 }
								 }

								 if (images.empty()) {
									 log_tool(cfg, "ERROR", "No valid training images found");
									 return false;
								 }

								 FaceRecWrapper rec(method_l);

								 if (!rec.CreateRecognizer()) {
									 log_tool(cfg, "ERROR", "Recognizer creation failed");
									 return false;
								 }

								 if (!rec.Train(images, labels)) {
									 log_tool(cfg, "ERROR", "Training failed");
									 return false;
								 }

								 std::string model_out =
								 outputModel.empty() ? fa_user_model_path(cfg, user) : outputModel;

								 if (!rec.Save(model_out)) {
									 log_tool(cfg, "ERROR", "Cannot save model: %s", model_out.c_str());
		return false;
								 }

	log_tool(cfg, "INFO", "Model saved to %s", model_out.c_str());
	return true;
							 }

// ==========================================================
// TEST USER (SFace se disponibile, altrimenti LBPH/Eigen/Fisher)
// ==========================================================

bool fa_test_user(const std::string &user,
				  const FacialAuthConfig &cfg,
				  const std::string &modelPath,
				  double &best_conf,
				  int &best_label,
				  std::string &logbuf,
				  double threshold_override)
{
	best_conf  = 1e9;
	best_label = -1;

	// ===== pipeline SFace =====
	std::string emb_file = fa_user_embedding_path(cfg, user);
	bool have_sface =
		!cfg.sface_model.empty() &&
		file_exists(cfg.sface_model) &&
		file_exists(emb_file);

		if (have_sface) {
			cv::Mat emb_ref;
			if (!fa_load_embedding_binary(emb_file, emb_ref) || emb_ref.empty()) {
				log_tool(cfg, "ERROR", "Cannot load SFace embedding: %s",
						 emb_file.c_str());
				return false;
			}

			// backend/target: se detector_profile=yunet_cuda proviamo CUDA, altrimenti CPU
			std::string detp = cfg.detector_profile;
			for (char &c : detp)
				c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

			bool want_cuda = (detp == "yunet_cuda");

			int backend_id = cv::dnn::DNN_BACKEND_OPENCV;
			int target_id  = cv::dnn::DNN_TARGET_CPU;

			if (want_cuda) {
				backend_id = cv::dnn::DNN_BACKEND_CUDA;
				target_id  = cv::dnn::DNN_TARGET_CUDA;
			}

			cv::Ptr<cv::FaceRecognizerSF> sface;
			try {
				sface = cv::FaceRecognizerSF::create(cfg.sface_model, "",
													 backend_id, target_id);
			} catch (...) {
				if (want_cuda) {
					log_tool(cfg, "WARN",
							 "SFace CUDA init failed, falling back to CPU");
					try {
						sface = cv::FaceRecognizerSF::create(cfg.sface_model, "",
															 cv::dnn::DNN_BACKEND_OPENCV,
										   cv::dnn::DNN_TARGET_CPU);
					} catch (...) {
						log_tool(cfg, "ERROR",
								 "Cannot create SFace recognizer on CPU");
						return false;
					}
				} else {
					log_tool(cfg, "ERROR",
							 "Cannot create SFace recognizer");
					return false;
				}
			}

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

			double threshold = (threshold_override > 0.0)
			? threshold_override
			: cfg.sface_threshold;

			if (threshold <= 0.0)
				threshold = 0.50;

			// Rispetta detector_profile anche qui per il detector
			std::string det = cfg.detector_profile;
			for (char &c : det)
				c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

			bool want_yunet = (det == "yunet_cpu" || det == "yunet_cuda");

			cv::Ptr<cv::FaceDetectorYN> yunet;
			bool yunet_ok = want_yunet &&
			!cfg.yunet_model.empty() &&
			file_exists(cfg.yunet_model);

			FaceRecWrapper haar_det("lbph");
			bool use_haar = false;

			if (!cfg.haar_cascade_path.empty() && file_exists(cfg.haar_cascade_path)) {
				if (haar_det.InitCascade(cfg.haar_cascade_path))
					use_haar = true;
			}

			cv::Mat frame;
			for (int i = 0; i < cfg.frames; i++) {
				cap >> frame;
				if (frame.empty())
					continue;

				cv::Rect roi;
				bool detected = false;

				if (yunet_ok) {
					try {
						if (yunet.empty()) {
							bool want_cuda_det = (det == "yunet_cuda");

							int be_id = cv::dnn::DNN_BACKEND_OPENCV;
							int ta_id = cv::dnn::DNN_TARGET_CPU;

							if (want_cuda_det) {
								be_id = cv::dnn::DNN_BACKEND_CUDA;
								ta_id = cv::dnn::DNN_TARGET_CUDA;
							}

							try {
								yunet = cv::FaceDetectorYN::create(
									cfg.yunet_model,
									"",
									frame.size(),
																   0.9f, 0.3f, 5000,
										   be_id,
										   ta_id
								);
							} catch (...) {
								if (want_cuda_det) {
									log_tool(cfg, "WARN",
											 "YUNet CUDA init failed in test, fallback CPU");
									try {
										yunet = cv::FaceDetectorYN::create(
											cfg.yunet_model,
											 "",
											 frame.size(),
																		   0.9f, 0.3f, 5000,
											 cv::dnn::DNN_BACKEND_OPENCV,
											 cv::dnn::DNN_TARGET_CPU
										);
									} catch (...) {
										yunet.release();
									}
								} else {
									yunet.release();
								}
							}
						}

						if (!yunet.empty()) {
							yunet->setInputSize(frame.size());
							cv::Mat faces;
							yunet->detect(frame, faces);

							if (!faces.empty() && faces.rows > 0) {
								const float *row = faces.ptr<float>(0);
								float x = row[0];
								float y = row[1];
								float w = row[2];
								float h = row[3];

								roi = cv::Rect(
									cv::Point(cvRound(x), cvRound(y)),
											   cv::Size(cvRound(w), cvRound(h))
								);
								roi &= cv::Rect(0, 0, frame.cols, frame.rows);
								if (roi.width > 0 && roi.height > 0)
									detected = true;
							}
						}
					} catch (...) {
						log_tool(cfg, "WARN",
								 "YUNet detection failed in test, fallback HAAR");
						yunet_ok = false;
					}
				}

				if (!detected && use_haar) {
					if (haar_det.DetectFace(frame, roi))
						detected = true;
				}

				if (!detected)
					continue;

				cv::Mat face = frame(roi).clone();
				if (face.empty())
					continue;

				cv::Mat resized;
				cv::resize(face, resized, cv::Size(112, 112));

				cv::Mat emb_live;
				try {
					sface->feature(resized, emb_live);
				} catch (...) {
					continue;
				}

				if (emb_live.empty())
					continue;

				double dist = fa_cosine_distance(emb_ref, emb_live);

				if (dist < best_conf)
					best_conf = dist;

				best_label = 0;

				if (dist <= threshold) {
					log_tool(cfg, "INFO",
							 "Auth success (model=sface): dist=%.3f <= %.3f",
							 dist, threshold);
					return true;
				}

				sleep_ms(cfg.sleep_ms);
			}

			log_tool(cfg, "WARN",
					 "Auth failed (model=sface): best_dist=%.3f threshold=%.3f",
					 best_conf, threshold);
			return false;
		}

		// ===== pipeline classica LBPH/Eigen/Fisher =====

		std::string model_file =
		modelPath.empty() ? fa_user_model_path(cfg, user) : modelPath;

		if (!file_exists(model_file)) {
			log_tool(cfg, "ERROR", "Model file missing: %s", model_file.c_str());
		return false;
		}

	// Determina il tipo modello dal file
	std::string model_type = fa_detect_model_type(model_file);

	FaceRecWrapper rec(model_type);

	if (!rec.CreateRecognizer()) {
		log_tool(cfg, "ERROR", "Recognizer creation failed (%s)",
				 model_type.c_str());
		return false;
	}

	if (!rec.Load(model_file)) {
		log_tool(cfg, "ERROR", "Cannot load model: %s", model_file.c_str());
		return false;
	}

	// Carichiamo la HAAR cascade come in capture()
	if (cfg.haar_cascade_path.empty() || !file_exists(cfg.haar_cascade_path)) {
		log_tool(cfg, "ERROR",
				 "haar_cascade_path is missing or invalid in config");
		return false;
	}

	if (!rec.InitCascade(cfg.haar_cascade_path)) {
		log_tool(cfg, "ERROR", "Cannot load HAAR cascade from %s",
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

	log_tool(cfg, "INFO", "Testing user %s on device %s",
			 user.c_str(), dev_used.c_str());

	best_conf  = 1e9;
	best_label = -1;

	// Soglia per metodo
	double threshold = cfg.lbph_threshold;
	if (model_type == "eigen")  threshold = cfg.eigen_threshold;
	if (model_type == "fisher") threshold = cfg.fisher_threshold;

	if (threshold_override > 0.0)
		threshold = threshold_override;

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

		if (gray.cols < 60 || gray.rows < 60)
			continue;

		// *** Resize uniforme per tutti i metodi ***
		cv::resize(gray, gray, cv::Size(200, 200));

		int    label = -1;
		double conf  = 1e9;

		if (!rec.Predict(gray, label, conf))
			continue;

		if (conf < best_conf) {
			best_conf  = conf;
			best_label = label;
		}

		if (conf <= threshold) {
			log_tool(cfg, "INFO",
					 "Auth success (model=%s): conf=%.2f <= %.2f",
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
	<< "      --detector NAME     Face detector: haar|yunet_cpu|yunet_cuda\n"
	<< "      --clean             Remove all user images\n"
	<< "      --reset             Remove user model + images\n"
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
	"  -m, --method <type>         Training method (lbph, eigen, fisher, sface)\n"
	"  -i, --input <basedir>       Override basedir for images/models\n"
	"  -o, --output <file>         Path to save the trained model (XML or embedding)\n"
	"  -f, --force                 Force overwrite (non usato internamente)\n"
	"  -v, --verbose               Enable detailed output\n"
	"  -h, --help                  Show this help message\n";
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

	std::string method_l = method;
	for (char &c : method_l)
		c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

	if (method_l != "lbph" && method_l != "eigen" &&
		method_l != "fisher" && method_l != "sface") {
		std::cerr << "ERROR: Invalid method '" << method << "'\n";
	return 1;
		}

		if (!fa_check_root("facial_training"))
			return 1;

	if (!input_dir.empty())
		cfg.basedir = input_dir;

	if (output_file.empty())
		output_file = (method_l == "sface")
		? fa_user_embedding_path(cfg, user)
		: fa_user_model_path(cfg, user);

	std::string train_dir = fa_user_image_dir(cfg, user);

	if (!fa_train_user(user, cfg, method_l, train_dir, output_file, force, logbuf)) {
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

static void print_facial_test_usage(const char *p)
{
	std::cout <<
	"Usage: facial_test -u <user> -m <model_path> [options]\n"
	"\n"
	"Options:\n"
	"  -u, --user <user>        Utente da verificare (obbligatorio)\n"
	"  -m, --model <path>       File modello XML o embedding (opzionale)\n"
	"  -c, --config <file>      File di configurazione\n"
	"                           (default: /etc/security/pam_facial.conf)\n"
	"  -d, --device <device>    Dispositivo webcam (es. /dev/video0)\n"
	"      --threshold <value>  Soglia di confidenza / distanza\n"
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

	// Config
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

	double best_conf = 0.0;
	int    best_label = -1;

	bool ok = fa_test_user(user, cfg, model_path,
						   best_conf, best_label, logbuf,
						threshold_override);

	if (!ok) {
		std::cerr << "Authentication FAILED (best_conf=" << best_conf << ")\n";
		if (!logbuf.empty()) std::cerr << logbuf;
		return 2;
	}

	std::cout << "[OK] Authentication SUCCESS (conf=" << best_conf << ")\n";
	return 0;
}
