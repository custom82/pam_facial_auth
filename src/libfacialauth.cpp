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
#include <chrono>
#include <thread>
#include <cctype>

namespace fs = std::filesystem;

// ==========================================================
// Utility helpers
// ==========================================================

static bool file_exists(const std::string &path) {
	std::error_code ec;
	return fs::exists(path, ec);
}

static void ensure_dirs(const std::string &path) {
	if (path.empty())
		return;
	std::error_code ec;
	fs::create_directories(path, ec);
}

static std::string join_path(const std::string &a, const std::string &b) {
	if (a.empty()) return b;
	if (b.empty()) return a;
	return (fs::path(a) / b).string();
}

static std::string trim(const std::string &s) {
	const char *ws = " \t\r\n";
	auto start = s.find_first_not_of(ws);
	if (start == std::string::npos)
		return {};
	auto end = s.find_last_not_of(ws);
	return s.substr(start, end - start + 1);
}

static std::string to_lower_str(const std::string &s) {
	std::string out;
	out.reserve(s.size());
	for (unsigned char c : s)
		out.push_back(static_cast<char>(std::tolower(c)));
	return out;
}

static void sleep_ms(int ms) {
	if (ms <= 0) return;
	std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

static void log_append(std::string &log, const std::string &line) {
	log.append(line);
	log.push_back('\n');
}

static void log_tool(const FacialAuthConfig &cfg,
					 const char *level,
					 const char *fmt, ...)
{
	char buf[1024];
	va_list ap;
	va_start(ap, fmt);
	vsnprintf(buf, sizeof(buf), fmt, ap);
	va_end(ap);

	std::string msg = std::string("[") + level + "] " + buf;
	if (cfg.debug) {
		std::cerr << msg << std::endl;
	}
}

// ==========================================================
// Config
// ==========================================================

bool fa_load_config(const std::string &path,
					FacialAuthConfig &cfg,
					std::string &log)
{
	cfg.config_path = path;

	if (!file_exists(path)) {
		log_append(log, "Config file not found, using defaults: " + path);
		return false;
	}

	std::ifstream in(path);
	if (!in) {
		log_append(log, "Failed to open config file: " + path);
		return false;
	}

	std::string line;
	while (std::getline(in, line)) {
		line = trim(line);
		if (line.empty() || line[0] == '#')
			continue;
		auto pos = line.find('=');
		if (pos == std::string::npos)
			continue;
		std::string key = trim(line.substr(0, pos));
		std::string val = trim(line.substr(pos + 1));
		std::string k   = to_lower_str(key);

		if (k == "basedir")              cfg.basedir = val;
		else if (k == "device")          cfg.device = val;
		else if (k == "camera_index")    cfg.camera_index = std::stoi(val);
		else if (k == "frames")          cfg.frames = std::stoi(val);
		else if (k == "width")           cfg.width  = std::stoi(val);
		else if (k == "height")          cfg.height = std::stoi(val);
		else if (k == "frame_width")     cfg.width  = std::stoi(val);
		else if (k == "frame_height")    cfg.height = std::stoi(val);
		else if (k == "sleep_ms")        cfg.sleep_ms = std::stoi(val);
		else if (k == "threshold")       cfg.threshold = std::stod(val);
		else if (k == "debug")           cfg.debug = (val == "1" || to_lower_str(val) == "true");
		else if (k == "force_overwrite") cfg.force_overwrite = (val == "1" || to_lower_str(val) == "true");

		// DNN generico
		else if (k == "dnn_type")        cfg.dnn_type = to_lower_str(val);
		else if (k == "dnn_model")       cfg.dnn_model_path = val;
		else if (k == "dnn_proto")       cfg.dnn_proto_path = val;
		else if (k == "dnn_device")      cfg.dnn_device = to_lower_str(val);
		else if (k == "dnn_threshold")   cfg.dnn_threshold = std::stod(val);
		else if (k == "dnn_profile")     cfg.dnn_profile = to_lower_str(val);

		// DNN modelli specifici
		else if (k == "dnn_model_fast")                 cfg.dnn_model_fast = val;
		else if (k == "dnn_model_sface")                cfg.dnn_model_sface = val;
		else if (k == "dnn_model_lresnet100")           cfg.dnn_model_lresnet100 = val;
		else if (k == "dnn_model_openface")             cfg.dnn_model_openface = val;

		else if (k == "dnn_model_yunet")                cfg.dnn_model_yunet = val;
		else if (k == "dnn_model_detector_caffe")       cfg.dnn_model_detector_caffe = val;
		else if (k == "dnn_model_detector_fp16")        cfg.dnn_model_detector_fp16 = val;
		else if (k == "dnn_model_detector_uint8")       cfg.dnn_model_detector_uint8 = val;
		else if (k == "dnn_proto_detector_caffe")       cfg.dnn_proto_detector_caffe = val;

		else if (k == "dnn_model_emotion")              cfg.dnn_model_emotion = val;
		else if (k == "dnn_model_keypoints")            cfg.dnn_model_keypoints = val;
		else if (k == "dnn_model_face_landmark_tflite") cfg.dnn_model_face_landmark_tflite = val;
		else if (k == "dnn_model_face_detection_tflite")cfg.dnn_model_face_detection_tflite = val;
		else if (k == "dnn_model_face_blendshapes_tflite") cfg.dnn_model_face_blendshapes_tflite = val;
	}

	log_append(log, "Config loaded from: " + path);
	return true;
}

std::string fa_user_image_dir(const FacialAuthConfig &cfg, const std::string &user) {
	return join_path(join_path(cfg.basedir, "images"), user);
}

std::string fa_user_model_path(const FacialAuthConfig &cfg, const std::string &user) {
	return join_path(join_path(cfg.basedir, "models"), user + ".xml");
}

// ==========================================================
// DNN profile selector
// ==========================================================

bool fa_select_dnn_profile(FacialAuthConfig &cfg,
						   const std::string &profile,
						   std::string &log)
{
	std::string p = to_lower_str(profile);
	cfg.dnn_profile = p;

	auto set = [&](const std::string &type,
				   const std::string &model_path,
				const std::string &proto_path)
	{
		cfg.dnn_type = type;
		cfg.dnn_model_path = model_path;
		cfg.dnn_proto_path = proto_path;
	};

	// Riconoscimento volto
	if (p == "fast" || p == "face_recognizer_fast") {
		std::string mp = !cfg.dnn_model_fast.empty()
		? cfg.dnn_model_fast
		: "/usr/share/opencv4/dnn/models/facial/face_recognizer_fast.onnx";
		set("onnx", mp, "");
	}
	else if (p == "sface") {
		std::string mp = !cfg.dnn_model_sface.empty()
		? cfg.dnn_model_sface
		: "/usr/share/opencv4/dnn/models/facial/face_recognition_sface_2021dec.onnx";
		set("onnx", mp, "");
	}
	else if (p == "lresnet100" || p == "lresnet100e_ir") {
		std::string mp = !cfg.dnn_model_lresnet100.empty()
		? cfg.dnn_model_lresnet100
		: "/usr/share/opencv4/dnn/models/facial/LResNet100E_IR.onnx";
		set("onnx", mp, "");
	}
	else if (p == "openface") {
		std::string mp = !cfg.dnn_model_openface.empty()
		? cfg.dnn_model_openface
		: "/usr/share/opencv4/dnn/models/facial/openface_nn4.small2.v1.t7";
		set("torch", mp, "");
	}

	// Detector volto
	else if (p == "yunet") {
		std::string mp = !cfg.dnn_model_yunet.empty()
		? cfg.dnn_model_yunet
		: "/usr/share/opencv4/dnn/models/facial/yunet-202303.onnx";
		set("onnx", mp, "");
	}
	else if (p == "det_caffe") {
		std::string mp = !cfg.dnn_model_detector_caffe.empty()
		? cfg.dnn_model_detector_caffe
		: "/usr/share/opencv4/dnn/models/facial/opencv_face_detector.caffemodel";
		std::string pp = !cfg.dnn_proto_detector_caffe.empty()
		? cfg.dnn_proto_detector_caffe
		: "/usr/share/opencv4/samples/dnn/face_detector/deploy.prototxt";
		set("caffe", mp, pp);
	}
	else if (p == "det_fp16") {
		std::string mp = !cfg.dnn_model_detector_fp16.empty()
		? cfg.dnn_model_detector_fp16
		: "/usr/share/opencv4/dnn/models/facial/opencv_face_detector_fp16.caffemodel";
		std::string pp = !cfg.dnn_proto_detector_caffe.empty()
		? cfg.dnn_proto_detector_caffe
		: "/usr/share/opencv4/samples/dnn/face_detector/deploy.prototxt";
		set("caffe", mp, pp);
	}
	else if (p == "det_uint8") {
		std::string mp = !cfg.dnn_model_detector_uint8.empty()
		? cfg.dnn_model_detector_uint8
		: "/usr/share/opencv4/dnn/models/facial/opencv_face_detector_uint8.pb";
		set("tensorflow", mp, "");
	}

	// Emotion / keypoints / MediaPipe TFLite
	else if (p == "emotion") {
		std::string mp = !cfg.dnn_model_emotion.empty()
		? cfg.dnn_model_emotion
		: "/usr/share/opencv4/dnn/models/facial/emotion_ferplus.onnx";
		set("onnx", mp, "");
	}
	else if (p == "keypoints") {
		std::string mp = !cfg.dnn_model_keypoints.empty()
		? cfg.dnn_model_keypoints
		: "/usr/share/opencv4/dnn/models/facial/facial_keypoints.onnx";
		set("onnx", mp, "");
	}
	else if (p == "mp_landmark") {
		std::string mp = !cfg.dnn_model_face_landmark_tflite.empty()
		? cfg.dnn_model_face_landmark_tflite
		: "/usr/share/opencv4/dnn/models/facial/face_landmark.tflite";
		set("tflite", mp, "");
	}
	else if (p == "mp_face") {
		std::string mp = !cfg.dnn_model_face_detection_tflite.empty()
		? cfg.dnn_model_face_detection_tflite
		: "/usr/share/opencv4/dnn/models/facial/face_detection_short_range.tflite";
		set("tflite", mp, "");
	}
	else if (p == "mp_blend") {
		std::string mp = !cfg.dnn_model_face_blendshapes_tflite.empty()
		? cfg.dnn_model_face_blendshapes_tflite
		: "/usr/share/opencv4/dnn/models/facial/face_blendshapes.tflite";
		set("tflite", mp, "");
	}
	else {
		log_append(log, "Unknown dnn_profile: " + profile);
		return false;
	}

	log_append(log, "DNN profile selected: " + p +
	" (type=" + cfg.dnn_type +
	", model=" + cfg.dnn_model_path + ")");
	return true;
}

// ==========================================================
// Camera helpers
// ==========================================================

static bool open_camera(const FacialAuthConfig &cfg,
						cv::VideoCapture &cap,
						std::string &device_used)
{
	if (!cfg.device.empty()) {
		if (!cap.open(cfg.device)) {
			return false;
		}
		device_used = cfg.device;
	} else {
		if (!cap.open(cfg.camera_index)) {
			return false;
		}
		device_used = "camera_index:" + std::to_string(cfg.camera_index);
	}

	if (cfg.width > 0)
		cap.set(cv::CAP_PROP_FRAME_WIDTH, cfg.width);
	if (cfg.height > 0)
		cap.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.height);

	return true;
}

// ==========================================================
// DNN helpers (Caffe / TensorFlow / ONNX / OpenVINO / TFLite / Torch)
// ==========================================================

namespace {

	cv::dnn::Net fa_create_dnn_net(const std::string &dnn_type,
								   const std::string &model,
								const std::string &proto)
	{
		std::string t = to_lower_str(dnn_type);

		if (t == "caffe") {
			if (proto.empty())
				throw std::runtime_error("DNN(Caffe): proto path is empty");
			return cv::dnn::readNetFromCaffe(proto, model);
		}
		else if (t == "tensorflow" || t == "tf") {
			return proto.empty()
			? cv::dnn::readNetFromTensorflow(model)
			: cv::dnn::readNetFromTensorflow(model, proto);
		}
		else if (t == "onnx") {
			return cv::dnn::readNetFromONNX(model);
		}
		else if (t == "openvino" || t == "ir") {
			// OpenVINO IR: XML + BIN (passiamo entrambi, bin può stare in proto)
			return cv::dnn::readNet(model, proto);
		}
		else if (t == "tflite") {
			// Importer TFLite (modello singolo .tflite)
			return cv::dnn::readNet(model);
		}
		else if (t == "torch" || t == "t7") {
			// Modelli Torch .t7
			return cv::dnn::readNetFromTorch(model);
		}

		throw std::runtime_error("Unknown DNN type: " + dnn_type);
	}

	void fa_set_dnn_backend_and_target(cv::dnn::Net &net,
									   const std::string &dnn_device)
	{
		std::string dev = to_lower_str(dnn_device);

		if (dev == "cuda") {
			#ifdef HAVE_OPENCV_DNN_CUDA
			net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
			net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
			#else
			net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
			net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
			#endif
		}
		else if (dev == "openvino" || dev == "ie") {
			#ifdef CV_DNN_BACKEND_INFERENCE_ENGINE
			net.setPreferableBackend(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);
			net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
			#else
			net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
			net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
			#endif
		}
		else if (dev == "opencl" || dev == "ocl") {
			net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
			net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
		}
		else {
			net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
			net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
		}
	}

} // anon namespace

// ==========================================================
// FaceRecWrapper
// ==========================================================

FaceRecWrapper::FaceRecWrapper()
: FaceRecWrapper("lbph")
{
}

FaceRecWrapper::FaceRecWrapper(const std::string &modelType_)
: modelType(to_lower_str(modelType_))
{
	if (modelType == "lbph" || modelType.empty()) {
		recognizer = cv::face::LBPHFaceRecognizer::create();
	}
	else if (modelType == "eigen") {
		recognizer = cv::face::EigenFaceRecognizer::create();
	}
	else if (modelType == "fisher") {
		recognizer = cv::face::FisherFaceRecognizer::create();
	}
	else if (modelType == "dnn") {
		// Per DNN usiamo un riconoscitore LBPH "dummy" per avere un XML valido
		recognizer = cv::face::LBPHFaceRecognizer::create();
		use_dnn = true;
	}
	else {
		recognizer = cv::face::LBPHFaceRecognizer::create();
	}
}

void FaceRecWrapper::ConfigureDNN(const FacialAuthConfig &cfg)
{
	dnn_profile    = cfg.dnn_profile;
	dnn_type       = cfg.dnn_type;
	dnn_model_path = cfg.dnn_model_path;
	dnn_proto_path = cfg.dnn_proto_path;
	dnn_device     = cfg.dnn_device;
	dnn_threshold  = cfg.dnn_threshold;
	use_dnn        = true;

	try {
		dnn_net = fa_create_dnn_net(dnn_type, dnn_model_path, dnn_proto_path);
		fa_set_dnn_backend_and_target(dnn_net, dnn_device);
		dnn_loaded = true;
	} catch (const std::exception &e) {
		std::cerr << "Error loading DNN: " << e.what() << std::endl;
		dnn_loaded = false;
		use_dnn    = false;
	}
}

bool FaceRecWrapper::load_dnn_from_model_file(const std::string &modelFile)
{
	cv::FileStorage fs(modelFile, cv::FileStorage::READ);
	if (!fs.isOpened())
		return false;

	int enabled = 0;
	fs["fa_dnn_enabled"] >> enabled;

	std::string algorithm;
	fs["fa_algorithm"] >> algorithm;
	algorithm = to_lower_str(algorithm);

	if (algorithm != "dnn" || !enabled) {
		// Modello classico o DNN disabilitato
		use_dnn    = false;
		dnn_loaded = false;
		return false;
	}

	fs["fa_dnn_profile"]   >> dnn_profile;
	fs["fa_dnn_type"]      >> dnn_type;
	fs["fa_dnn_model"]     >> dnn_model_path;
	fs["fa_dnn_proto"]     >> dnn_proto_path;
	fs["fa_dnn_device"]    >> dnn_device;
	fs["fa_dnn_threshold"] >> dnn_threshold;

	if (dnn_threshold <= 0.0)
		dnn_threshold = 0.6;

	try {
		dnn_net = fa_create_dnn_net(dnn_type, dnn_model_path, dnn_proto_path);
		fa_set_dnn_backend_and_target(dnn_net, dnn_device);
		dnn_loaded = true;
		use_dnn    = true;
		return true;
	}
	catch (const std::exception &e) {
		std::cerr << "Error loading DNN from XML: " << e.what() << std::endl;
		dnn_loaded = false;
		use_dnn    = false;
		return false;
	}
}

bool FaceRecWrapper::Load(const std::string &modelFile) {
	try {
		recognizer->read(modelFile);
		// prova a leggere meta DNN / algoritmo
		load_dnn_from_model_file(modelFile);
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

		// Scriviamo sempre l'algoritmo usato + eventuali metadati DNN
		cv::FileStorage fs(modelFile, cv::FileStorage::APPEND);
		if (fs.isOpened()) {
			fs << "fa_algorithm" << modelType;

			if (use_dnn) {
				fs << "fa_dnn_enabled"   << 1;
				fs << "fa_dnn_profile"   << dnn_profile;
				fs << "fa_dnn_type"      << dnn_type;
				fs << "fa_dnn_model"     << dnn_model_path;
				fs << "fa_dnn_proto"     << dnn_proto_path;
				fs << "fa_dnn_device"    << dnn_device;
				fs << "fa_dnn_threshold" << dnn_threshold;
			}
		}

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

	// Per un DNN puro pre-addestrato possiamo saltare il training
	if (use_dnn)
		return true;

	try {
		recognizer->train(images, labels);
		return true;
	} catch (const std::exception &e) {
		std::cerr << "Error training model: " << e.what() << std::endl;
		return false;
	}
						   }

						   bool FaceRecWrapper::predict_with_dnn(const cv::Mat &faceGray,
																 int &label,
											   double &confidence)
						   {
							   if (!dnn_loaded)
								   return false;

							   cv::Mat input;
							   if (faceGray.channels() == 1)
								   cv::cvtColor(faceGray, input, cv::COLOR_GRAY2BGR);
							   else
								   input = faceGray;

							   // Per ora usiamo ancora la logica SSD-like:
							   const cv::Size inputSize(300, 300);
							   const double scaleFactor = 1.0;
							   const cv::Scalar meanVal(104.0, 177.0, 123.0);

							   cv::Mat blob = cv::dnn::blobFromImage(input, scaleFactor, inputSize,
																	 meanVal, false, false);
							   dnn_net.setInput(blob);
							   cv::Mat out = dnn_net.forward();

							   float best_score = 0.0f;

							   // Caso classico SSD: [1, 1, N, 7]
							   if (out.dims == 4 && out.size[2] > 0 && out.size[3] >= 7) {
								   int N = out.size[2];
								   const float* data = out.ptr<float>(0, 0); // [N][7]

								   for (int i = 0; i < N; ++i) {
									   float score = data[i * 7 + 2]; // confidence
									   if (score > best_score)
										   best_score = score;
								   }
							   }
							   // Fallback generico: [N,7]
							   else if (out.rows > 0 && out.cols >= 7) {
								   int N = out.rows;
								   for (int i = 0; i < N; ++i) {
									   const float* row = out.ptr<float>(i);
									   float score = row[2];
									   if (score > best_score)
										   best_score = score;
								   }
							   }

							   // conf più basso = migliore (coerente col resto del codice)
							   confidence = 1.0 - static_cast<double>(best_score);

							   if (best_score >= dnn_threshold)
								   label = 1;
							   else
								   label = -1;

							   return true;
						   }

						   bool FaceRecWrapper::Predict(const cv::Mat &faceGray,
														int &label,
									  double &confidence)
						   {
							   if (faceGray.empty())
								   return false;

							   if (use_dnn && dnn_loaded) {
								   return predict_with_dnn(faceGray, label, confidence);
							   }

							   try {
								   recognizer->predict(faceGray, label, confidence);
								   return true;
							   } catch (const std::exception &e) {
								   std::cerr << "Error predicting: " << e.what() << std::endl;
								   return false;
							   }
						   }

						   bool FaceRecWrapper::DetectFace(const cv::Mat &frame, cv::Rect &faceROI) {
							   if (frame.empty())
								   return false;

							   if (faceCascade.empty()) {
								   std::string cascadePath;
								   const char *envPath = std::getenv("FACIAL_HAAR_PATH");
								   if (envPath)
									   cascadePath = envPath;

								   if (cascadePath.empty())
									   cascadePath = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml";
								   if (!file_exists(cascadePath))
									   cascadePath = "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml";

								   if (!faceCascade.load(cascadePath)) {
									   std::cerr << "Failed to load Haar cascade: " << cascadePath << std::endl;
									   return false;
								   }
							   }

							   cv::Mat gray;
							   if (frame.channels() == 3)
								   cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
							   else
								   gray = frame;

							   cv::equalizeHist(gray, gray);

							   std::vector<cv::Rect> faces;
							   faceCascade.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(80, 80));

							   if (faces.empty())
								   return false;

							   auto best = faces[0];
							   for (const auto &r : faces) {
								   if (r.area() > best.area())
									   best = r;
							   }

							   faceROI = best;
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
							   (void)log;

							   std::string device_used;
							   cv::VideoCapture cap;
							   if (!open_camera(cfg, cap, device_used)) {
								   log_tool(cfg, "ERROR", "Failed to open camera (%s)", cfg.device.c_str());
		return false;
							   }
	log_tool(cfg, "INFO", "Camera opened on %s", device_used.c_str());

	std::string img_dir = fa_user_image_dir(cfg, user);
	ensure_dirs(img_dir);

	if (!force && !cfg.force_overwrite) {
		// Non cancelliamo immagini esistenti: continuiamo con indice alto
	}

	// Determina indice di partenza in base ai file esistenti
	int start_index = 0;
	int max_idx = 0;
	for (auto &entry : fs::directory_iterator(img_dir)) {
		if (!entry.is_regular_file()) continue;
		std::string fname = entry.path().filename().string();
		if (fname.rfind("img_", 0) == 0 && fname.size() >= 8) {
			try {
				int idx = std::stoi(fname.substr(4, 3));
				if (idx > max_idx)
					max_idx = idx;
			} catch (...) {
			}
		}
	}
	start_index = max_idx;

	log_tool(cfg, "INFO", "Existing max image index for user %s is %d",
			 user.c_str(), max_idx);

	FaceRecWrapper rec; // solo per DetectFace
	int captured = 0;

	std::string fmt = img_format.empty() ? "png" : img_format;
	for (auto &ch : fmt)
		ch = static_cast<char>(::tolower(static_cast<unsigned char>(ch)));

							   log_tool(cfg, "INFO", "Capturing %d frames", cfg.frames);

							   while (captured < cfg.frames) {
								   cv::Mat frame;
								   cap >> frame;
								   if (frame.empty())
									   break;

								   cv::Rect roi;
								   if (!rec.DetectFace(frame, roi)) {
									   // Nessun volto: skip frame
									   sleep_ms(cfg.sleep_ms);
									   continue;
								   }

								   cv::Mat face = frame(roi).clone();
								   cv::Mat gray;
								   cv::cvtColor(face, gray, cv::COLOR_BGR2GRAY);
								   cv::resize(gray, gray, cv::Size(cfg.width, cfg.height));
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

							   if (captured == 0) {
								   log_tool(cfg, "ERROR", "No frames captured for user %s", user.c_str());
		return false;
							   }

	return true;
						   }

bool fa_train_user(const std::string &user,
				   const FacialAuthConfig &cfg,
				   const std::string &method,
				   const std::string &inputDir,
				   const std::string &outputModel,
				   bool force,
				   std::string &log)
{
	(void)force;
	(void)log;

	std::string train_dir = inputDir.empty()
						  ? fa_user_image_dir(cfg, user)
						  : inputDir;

	if (!fs::exists(train_dir)) {
		log_tool(cfg, "ERROR", "Training dir does not exist: %s", train_dir.c_str());
		return false;
	}

	std::vector<cv::Mat> images;
	std::vector<int> labels;

	for (auto &entry : fs::directory_iterator(train_dir)) {
		if (!entry.is_regular_file()) continue;
		if (!entry.path().has_extension()) continue;

		std::string path = entry.path().string();
		cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
		if (!img.empty()) {
			cv::resize(img, img, cv::Size(cfg.width, cfg.height));
			images.push_back(img);
			labels.push_back(0); // single user: label fisso
		}
	}

	if (images.empty()) {
		log_tool(cfg, "ERROR", "No training images found in %s", train_dir.c_str());
		return false;
	}

	std::string mt = to_lower_str(method);
	FaceRecWrapper rec(mt);

	if (mt == "dnn") {
		rec.ConfigureDNN(cfg);
	}

	if (!rec.Train(images, labels)) {
		log_tool(cfg, "ERROR", "Training failed");
		return false;
	}

	std::string model_path = outputModel.empty()
						   ? fa_user_model_path(cfg, user)
						   : outputModel;

	if (!rec.Save(model_path)) {
		log_tool(cfg, "ERROR", "Failed to save model to %s", model_path.c_str());
		return false;
	}

	log_tool(cfg, "INFO", "Model saved to %s", model_path.c_str());
	return true;
}

bool fa_test_user(const std::string &user,
				  const FacialAuthConfig &cfg,
				  const std::string &modelPath,
				  double &best_conf,
				  int &best_label,
				  std::string &log)
{
	(void)log;

	std::string model_file = modelPath.empty()
						   ? fa_user_model_path(cfg, user)
						   : modelPath;

	if (!file_exists(model_file)) {
		log_tool(cfg, "ERROR", "Model file missing for user %s: %s",
				 user.c_str(), model_file.c_str());
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
	log_tool(cfg, "INFO", "Camera opened on %s", device_used.c_str());

	best_conf  = 1e9;
	best_label = -1;

	int frames_ok = 0;

	for (int i = 0; i < cfg.frames; ++i) {
		cv::Mat frame;
		cap >> frame;
		if (frame.empty())
			break;

		cv::Rect roi;
		if (!rec.DetectFace(frame, roi)) {
			log_tool(cfg, "DEBUG", "Frame %d: no face detected", i);
			sleep_ms(cfg.sleep_ms);
			continue;
		}

		cv::Mat face = frame(roi).clone();
		cv::Mat gray;
		cv::cvtColor(face, gray, cv::COLOR_BGR2GRAY);
		cv::resize(gray, gray, cv::Size(cfg.width, cfg.height));
		cv::equalizeHist(gray, gray);

		int label = -1;
		double conf = 0.0;
		if (rec.Predict(gray, label, conf)) {
			log_tool(cfg, "INFO", "Frame %d: label=%d conf=%.4f", i, label, conf);

			if (conf < best_conf) {
				best_conf  = conf;
				best_label = label;
			}
			++frames_ok;

			if (conf <= cfg.threshold) {
				log_tool(cfg, "INFO",
						 "Facial authentication SUCCESS (conf=%.4f <= thr=%.4f)",
						 conf, cfg.threshold);
				return true;
			}
		}

		sleep_ms(cfg.sleep_ms);
	}

	log_tool(cfg, "WARNING",
			 "Facial authentication FAILED for user %s (best_conf=%.4f thr=%.4f)",
			 user.c_str(), best_conf, cfg.threshold);

	return false;
}

// ==========================================================
// Maintenance
// ==========================================================

bool fa_clean_images(const FacialAuthConfig &cfg, const std::string &user)
{
	std::string img_dir = fa_user_image_dir(cfg, user);
	if (!fs::exists(img_dir))
		return true;

	std::error_code ec;
	for (auto &entry : fs::directory_iterator(img_dir)) {
		if (entry.is_regular_file()) {
			fs::remove(entry.path(), ec);
		}
	}
	return true;
}

bool fa_clean_model(const FacialAuthConfig &cfg, const std::string &user)
{
	std::string model = fa_user_model_path(cfg, user);
	if (!file_exists(model))
		return true;
	std::error_code ec;
	fs::remove(model, ec);
	return true;
}

void fa_list_images(const FacialAuthConfig &cfg, const std::string &user)
{
	std::string img_dir = fa_user_image_dir(cfg, user);
	if (!fs::exists(img_dir)) {
		std::cout << "No images dir for user " << user << ": " << img_dir << "\n";
		return;
	}

	std::cout << "Images for user " << user << " in " << img_dir << ":\n";
	for (auto &entry : fs::directory_iterator(img_dir)) {
		if (entry.is_regular_file())
			std::cout << "  " << entry.path().filename().string() << "\n";
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
