#include "../include/libfacialauth.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cstdarg>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <sys/stat.h>
#include <unistd.h>
#include <chrono>
#include <thread>
#include <cctype>
#include <getopt.h>
#include <algorithm>

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

		// DNN generic
		else if (k == "dnn_type")        cfg.dnn_type = to_lower_str(val);
		else if (k == "dnn_model")       cfg.dnn_model_path = val;
		else if (k == "dnn_proto")       cfg.dnn_proto_path = val;
		else if (k == "dnn_device")      cfg.dnn_device = to_lower_str(val);
		else if (k == "dnn_threshold")   cfg.dnn_threshold = std::stod(val);
		else if (k == "dnn_profile")     cfg.dnn_profile = to_lower_str(val);

		// Detector-specific
		else if (k == "detector_profile")   cfg.detector_profile = to_lower_str(val);
		else if (k == "detector_threshold") cfg.detector_threshold = std::stod(val);
		else if (k == "haar_cascade")       cfg.haar_cascade = val;

		// DNN models (recognition)
		else if (k == "dnn_model_fast")          cfg.dnn_model_fast = val;
		else if (k == "dnn_model_sface")         cfg.dnn_model_sface = val;
		else if (k == "dnn_model_lresnet100")    cfg.dnn_model_lresnet100 = val;
		else if (k == "dnn_model_openface")      cfg.dnn_model_openface = val;

		// DNN models (detectors)
		else if (k == "dnn_model_yunet")               cfg.dnn_model_yunet = val;
		else if (k == "dnn_model_detector_caffe")      cfg.dnn_model_detector_caffe = val;
		else if (k == "dnn_model_detector_fp16")       cfg.dnn_model_detector_fp16 = val;
		else if (k == "dnn_model_detector_uint8")      cfg.dnn_model_detector_uint8 = val;
		else if (k == "dnn_proto_detector_caffe")      cfg.dnn_proto_detector_caffe = val;

		// DNN models (emotion / keypoints / MediaPipe)
		else if (k == "dnn_model_emotion")                 cfg.dnn_model_emotion = val;
		else if (k == "dnn_model_keypoints")               cfg.dnn_model_keypoints = val;
		else if (k == "dnn_model_face_landmark_tflite")    cfg.dnn_model_face_landmark_tflite = val;
		else if (k == "dnn_model_face_detection_tflite")   cfg.dnn_model_face_detection_tflite = val;
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

	// Recognition
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
	// Detector / misc (we still configure dnn_* generics)
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
			// OpenVINO IR: XML + BIN (BIN can be passed via proto)
			return cv::dnn::readNet(model, proto);
		}
		else if (t == "tflite") {
			// TFLite importer (single .tflite file)
			return cv::dnn::readNet(model);
		}
		else if (t == "torch" || t == "t7") {
			// Torch .t7 models
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

} // anonymous namespace

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
		// For DNN we still keep a dummy LBPH to produce a valid XML
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
	if (cfg.dnn_threshold > 0.0)
		dnn_threshold = cfg.dnn_threshold;
	else
		dnn_threshold = 0.6;

	use_dnn        = true;
	dnn_loaded     = false;

	try {
		dnn_net = fa_create_dnn_net(dnn_type, dnn_model_path, dnn_proto_path);
		fa_set_dnn_backend_and_target(dnn_net, dnn_device);
		dnn_loaded = true;
	} catch (const std::exception &e) {
		std::cerr << "Error loading DNN recognizer: " << e.what() << std::endl;
		dnn_loaded = false;
		use_dnn    = false;
	}
}

void FaceRecWrapper::ConfigureDetector(const FacialAuthConfig &cfg)
{
	detector_profile   = cfg.detector_profile;
	detector_threshold = (cfg.detector_threshold > 0.0)
	? cfg.detector_threshold
	: cfg.dnn_threshold;
	if (detector_threshold <= 0.0)
		detector_threshold = 0.6;

	haar_cascade_path = cfg.haar_cascade;

	use_dnn_detector = false;
	dnn_detector_net = cv::dnn::Net();

	std::string p = to_lower_str(detector_profile);
	std::string type, model, proto;

	if (p == "det_caffe") {
		model = !cfg.dnn_model_detector_caffe.empty()
		? cfg.dnn_model_detector_caffe
		: "/usr/share/opencv4/dnn/models/facial/opencv_face_detector.caffemodel";
		proto = !cfg.dnn_proto_detector_caffe.empty()
		? cfg.dnn_proto_detector_caffe
		: "/usr/share/opencv4/samples/dnn/face_detector/deploy.prototxt";
		type  = "caffe";
	}
	else if (p == "det_fp16") {
		model = !cfg.dnn_model_detector_fp16.empty()
		? cfg.dnn_model_detector_fp16
		: "/usr/share/opencv4/dnn/models/facial/opencv_face_detector_fp16.caffemodel";
		proto = !cfg.dnn_proto_detector_caffe.empty()
		? cfg.dnn_proto_detector_caffe
		: "/usr/share/opencv4/samples/dnn/face_detector/deploy.prototxt";
		type  = "caffe";
	}
	else if (p == "det_uint8") {
		model = !cfg.dnn_model_detector_uint8.empty()
		? cfg.dnn_model_detector_uint8
		: "/usr/share/opencv4/dnn/models/facial/opencv_face_detector_uint8.pb";
		type  = "tensorflow";
		proto.clear();
	}
	else if (p == "yunet") {
		// YuNet ONNX detector – we still use the generic DNN API,
		// but if something goes wrong we'll just fall back to Haar.
		model = !cfg.dnn_model_yunet.empty()
		? cfg.dnn_model_yunet
		: "/usr/share/opencv4/dnn/models/facial/yunet-202303.onnx";
		type  = "onnx";
		proto.clear();
	}

	if (!type.empty() && !model.empty()) {
		try {
			dnn_detector_net = fa_create_dnn_net(type, model, proto);
			fa_set_dnn_backend_and_target(dnn_detector_net, cfg.dnn_device);
			use_dnn_detector = true;
		} catch (const std::exception &e) {
			std::cerr << "Error loading DNN detector: " << e.what() << std::endl;
			use_dnn_detector = false;
		}
	}

	// If haar_cascade_path is empty we decide it lazily in DetectFace()
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
		// Classic model or DNN disabled
		use_dnn        = false;
		dnn_loaded     = false;
		has_dnn_template = false;
		return false;
	}

	fs["fa_dnn_profile"]   >> dnn_profile;
	fs["fa_dnn_type"]      >> dnn_type;
	fs["fa_dnn_model"]     >> dnn_model_path;
	fs["fa_dnn_proto"]     >> dnn_proto_path;
	fs["fa_dnn_device"]    >> dnn_device;
	fs["fa_dnn_threshold"] >> dnn_threshold;

	cv::Mat tpl;
	fs["fa_dnn_template"] >> tpl;
	if (!tpl.empty()) {
		tpl.convertTo(dnn_template, CV_32F);
		has_dnn_template = true;
	} else {
		has_dnn_template = false;
	}

	if (dnn_proto_path == "\"\"" || dnn_proto_path == "''")
		dnn_proto_path.clear();

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

bool FaceRecWrapper::Load(const std::string &modelFile)
{
	try {
		recognizer->read(modelFile);
		// Try to read DNN meta
		load_dnn_from_model_file(modelFile);
		return true;
	} catch (const std::exception &e) {
		std::cerr << "Error loading model: " << e.what() << std::endl;
		return false;
	}
}

bool FaceRecWrapper::Save(const std::string &modelFile) const
{
	try {
		ensure_dirs(fs::path(modelFile).parent_path().string());
		recognizer->write(modelFile);

		// Append metadata for algorithm + DNN
		cv::FileStorage fs(modelFile, cv::FileStorage::APPEND);
		if (fs.isOpened()) {
			fs << "fa_algorithm" << modelType;

			if (use_dnn) {
				fs << "fa_dnn_enabled"   << 1;
				fs << "fa_dnn_profile"   << dnn_profile;
				fs << "fa_dnn_type"      << dnn_type;
				fs << "fa_dnn_model"     << dnn_model_path;

				if (dnn_proto_path.empty() ||
					dnn_proto_path == "\"\"" ||
					dnn_proto_path == "''")
				{
					fs << "fa_dnn_proto" << "";
				} else {
					fs << "fa_dnn_proto" << dnn_proto_path;
				}

				fs << "fa_dnn_device"    << dnn_device;
				fs << "fa_dnn_threshold" << dnn_threshold;

				if (!dnn_template.empty()) {
					cv::Mat tmp;
					dnn_template.convertTo(tmp, CV_32F);
					fs << "fa_dnn_template" << tmp;
				}
			}
		}

		return true;
	} catch (const std::exception &e) {
		std::cerr << "Error saving model: " << e.what() << std::endl;
		return false;
	}
}

// ----------------------------------------------------------
// compute_dnn_embedding: face (gray) -> normalized embedding
// ----------------------------------------------------------

bool FaceRecWrapper::compute_dnn_embedding(const cv::Mat &faceGray, cv::Mat &embedding)
{
	if (!dnn_loaded)
		return false;
	if (faceGray.empty())
		return false;

	cv::Mat input;
	if (faceGray.channels() == 1)
		cv::cvtColor(faceGray, input, cv::COLOR_GRAY2BGR);
	else
		input = faceGray;

	cv::Mat resized;
	cv::Size inputSize(112, 112);

	// For most embedding models (SFace, fast, LResNet, OpenFace)
	cv::resize(input, resized, inputSize);

	cv::Mat blob = cv::dnn::blobFromImage(resized,
										  1.0 / 255.0,
									   inputSize,
									   cv::Scalar(0, 0, 0),
										  false,  // BGR
									   false);

	dnn_net.setInput(blob);
	cv::Mat out = dnn_net.forward();
	if (out.total() == 0)
		return false;

	cv::Mat flat = out.reshape(1, 1);
	flat.convertTo(embedding, CV_32F);
	if (embedding.cols == 0)
		return false;

	cv::normalize(embedding, embedding);
	return true;
}

bool FaceRecWrapper::Train(const std::vector<cv::Mat> &images,
						   const std::vector<int> &labels)
{
	if (images.empty() || labels.empty() || images.size() != labels.size())
		return false;

	if (use_dnn) {
		if (!dnn_loaded)
			return false;

		std::vector<cv::Mat> embs;
		embs.reserve(images.size());

		for (const auto &img : images) {
			cv::Mat emb;
			if (!compute_dnn_embedding(img, emb))
				continue;
			embs.push_back(emb);
		}

		if (embs.empty()) {
			std::cerr << "Error: no DNN embeddings computed during training.\n";
			return false;
		}

		int dim = embs[0].cols;
		cv::Mat mean = cv::Mat::zeros(1, dim, CV_32F);
		for (const auto &e : embs) {
			CV_Assert(e.cols == dim);
			mean += e;
		}
		mean /= static_cast<float>(embs.size());
		cv::normalize(mean, mean);

		dnn_template     = mean;
		has_dnn_template = true;

		// No classic recognizer training needed
		return true;
	}

	try {
		recognizer->train(images, labels);
		return true;
	} catch (const std::exception &e) {
		std::cerr << "Error training model: " << e.what() << std::endl;
		return false;
	}
}

// ==========================================================
// DNN prediction / detector
// ==========================================================

bool FaceRecWrapper::predict_with_dnn(const cv::Mat &faceGray,
									  int &label,
									  double &confidence)
{
	if (!dnn_loaded)
		return false;

	std::string p = to_lower_str(dnn_profile);

	// Embedding-based recognition: SFace, fast, LResNet, OpenFace
	if (p == "sface" || p == "fast" ||
		p == "lresnet100" || p == "openface")
	{
		if (!has_dnn_template || dnn_template.empty()) {
			label = -1;
			confidence = 1.0;
			return false;
		}

		cv::Mat emb;
		if (!compute_dnn_embedding(faceGray, emb)) {
			label = -1;
			confidence = 1.0;
			return false;
		}

		double sim = dnn_template.dot(emb);
		double dist = 1.0 - sim;   // cosine distance

		confidence = dist;         // lower is better
		if (dist <= dnn_threshold) {
			label = 1;
		} else {
			label = -1;
		}
		return true;
	}

	// If profile not recognized as embedding profile, just fail
	label = -1;
	confidence = 1.0;
	return false;
}

// SSD-like DNN detector: returns true if a face is found
// and fills faceROI with the (largest) detection.
bool FaceRecWrapper::dnn_detector_accepts(const cv::Mat &frame, cv::Rect &faceROI)
{
	if (!use_dnn_detector)
		return false;
	if (frame.empty())
		return false;

	cv::Mat input;
	if (frame.channels() == 1)
		cv::cvtColor(frame, input, cv::COLOR_GRAY2BGR);
	else
		input = frame;

	// Standard SSD face detector expects 300x300 & BGR mean
	cv::Size inputSize(300, 300);
	cv::Mat blob = cv::dnn::blobFromImage(input,
										  1.0,
									   inputSize,
									   cv::Scalar(104.0, 177.0, 123.0),
										  false, false);

	dnn_detector_net.setInput(blob);
	cv::Mat out = dnn_detector_net.forward();

	float best_score = 0.0f;
	cv::Rect best_rect;

	// Expected shape: [1,1,N,7] or [N,7]
	if (out.dims == 4 && out.size[3] >= 7) {
		int N = out.size[2];
		const float *data = out.ptr<float>(0, 0);
		for (int i = 0; i < N; ++i) {
			float score = data[i * 7 + 2];
			if (score < detector_threshold)
				continue;

			float x1 = data[i * 7 + 3];
			float y1 = data[i * 7 + 4];
			float x2 = data[i * 7 + 5];
			float y2 = data[i * 7 + 6];

			int fx1 = static_cast<int>(x1 * frame.cols);
			int fy1 = static_cast<int>(y1 * frame.rows);
			int fx2 = static_cast<int>(x2 * frame.cols);
			int fy2 = static_cast<int>(y2 * frame.rows);

			cv::Rect r(cv::Point(fx1, fy1), cv::Point(fx2, fy2));
			if (r.area() <= 0)
				continue;

			if (score > best_score) {
				best_score = score;
				best_rect  = r;
			}
		}
	}
	else if (out.rows > 0 && out.cols >= 7) {
		int N = out.rows;
		for (int i = 0; i < N; ++i) {
			const float *row = out.ptr<float>(i);
			float score = row[2];
			if (score < detector_threshold)
				continue;

			float x1 = row[3];
			float y1 = row[4];
			float x2 = row[5];
			float y2 = row[6];

			int fx1 = static_cast<int>(x1 * frame.cols);
			int fy1 = static_cast<int>(y1 * frame.rows);
			int fx2 = static_cast<int>(x2 * frame.cols);
			int fy2 = static_cast<int>(y2 * frame.rows);

			cv::Rect r(cv::Point(fx1, fy1), cv::Point(fy1, fy2));
			if (r.area() <= 0)
				continue;

			if (score > best_score) {
				best_score = score;
				best_rect  = r;
			}
		}
	}

	if (best_score >= detector_threshold && best_rect.area() > 0) {
		faceROI = best_rect;
		return true;
	}

	return false;
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

bool FaceRecWrapper::DetectFace(const cv::Mat &frame, cv::Rect &faceROI)
{
	if (frame.empty())
		return false;

	// 1) Try DNN detector if configured
	if (use_dnn_detector && !dnn_detector_net.empty()) {
		if (dnn_detector_accepts(frame, faceROI))
			return true;
	}

	// 2) Fallback to Haar cascade
	if (faceCascade.empty()) {
		std::string cascadePath = haar_cascade_path;

		if (cascadePath.empty()) {
			const char *envPath = std::getenv("FACIAL_HAAR_PATH");
			if (envPath)
				cascadePath = envPath;
		}

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

	if (force || cfg.force_overwrite) {
		std::error_code ec;
		for (auto &entry : fs::directory_iterator(img_dir)) {
			if (entry.is_regular_file())
				fs::remove(entry.path(), ec);
		}
	}

	// Determine starting index based on existing files
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

	FaceRecWrapper rec;
	rec.ConfigureDetector(cfg);

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
			// No face detected: skip frame
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
			labels.push_back(0); // single-user: fixed label
		}
	}

	if (images.empty()) {
		log_tool(cfg, "ERROR", "No training images found in %s", train_dir.c_str());
		return false;
	}

	std::string mt = to_lower_str(method);
	FaceRecWrapper rec(mt);

	if (mt == "dnn") {
		// Configure recognition DNN according to cfg
		FacialAuthConfig cfg_copy = cfg;
		// Ensure dnn_profile is set; if empty fallback to "sface"
		if (cfg_copy.dnn_profile.empty())
			cfg_copy.dnn_profile = "sface";
		fa_select_dnn_profile(cfg_copy, cfg_copy.dnn_profile, log);
		rec.ConfigureDNN(cfg_copy);
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

	// Configure detector based on cfg (DNN + Haar fallback)
	rec.ConfigureDetector(cfg);

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

	// Threshold selection: DNN vs classic
	double thr = cfg.threshold;
	if (rec.IsDNN()) {
		thr = rec.GetDnnThreshold();
		if (thr <= 0.0)
			thr = 0.6;
	}

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
			log_tool(cfg, "INFO", "Frame %d: label=%d conf=%.6f thr=%.6f",
					 i, label, conf, thr);

			if (label >= 0 && conf < best_conf) {
				best_conf  = conf;
				best_label = label;
			}

			++frames_ok;

			// conf più basso = meglio, e richiediamo label valido
			if (label >= 0 && conf <= thr && conf > 0.0) {
				log_tool(cfg, "INFO",
						 "Facial authentication SUCCESS (conf=%.6f <= thr=%.6f)",
						 conf, thr);
				return true;
			}
		}

		sleep_ms(cfg.sleep_ms);
	}

	if (best_label < 0) {
		log_tool(cfg, "WARNING",
				 "Facial authentication FAILED for user %s (no valid predictions)",
				 user.c_str());
		return false;
	}

	log_tool(cfg, "WARNING",
			 "Facial authentication FAILED for user %s (best_conf=%.6f thr=%.6f)",
			 user.c_str(), best_conf, thr);

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

// ======================================================================
// CLI helpers: dynamic DNN profiles, usage and entrypoints
// ======================================================================

// Read available DNN profiles from config (by presence of model paths)
static std::vector<std::string> fa_get_dnn_profiles_from_config(const std::string &config_path)
{
	std::vector<std::string> profiles;

	std::ifstream in(config_path);
	if (!in) {
		profiles = {
			"fast", "sface", "lresnet100", "openface",
			"yunet", "emotion", "keypoints",
			"det_uint8", "det_caffe", "det_fp16",
			"mp_landmark", "mp_face", "mp_blend"
		};
		return profiles;
	}

	auto maybe_add = [&](const std::string &line,
						 const std::string &key,
					  const std::string &name)
	{
		if (line.rfind(key, 0) != 0)
			return;

		auto eq = line.find('=');
		if (eq == std::string::npos)
			return;

		std::string val = line.substr(eq + 1);

		auto b = val.find_first_not_of(" \t");
		auto e = val.find_last_not_of(" \t\r\n");
		if (b == std::string::npos || e == std::string::npos)
			val.clear();
		else
			val = val.substr(b, e - b + 1);

		if (!val.empty())
			profiles.push_back(name);
	};

	std::string line;
	while (std::getline(in, line)) {
		auto first = line.find_first_not_of(" \t");
		if (first == std::string::npos)
			continue;
		if (line[first] == '#')
			continue;

		std::string s = line.substr(first);

		maybe_add(s, "dnn_model_fast",                     "fast");
		maybe_add(s, "dnn_model_sface",                    "sface");
		maybe_add(s, "dnn_model_lresnet100",               "lresnet100");
		maybe_add(s, "dnn_model_openface",                 "openface");

		maybe_add(s, "dnn_model_yunet",                    "yunet");
		maybe_add(s, "dnn_model_emotion",                  "emotion");
		maybe_add(s, "dnn_model_keypoints",                "keypoints");

		maybe_add(s, "dnn_model_detector_uint8",           "det_uint8");
		maybe_add(s, "dnn_model_detector_caffe",           "det_caffe");
		maybe_add(s, "dnn_model_detector_fp16",            "det_fp16");

		maybe_add(s, "dnn_model_face_landmark_tflite",     "mp_landmark");
		maybe_add(s, "dnn_model_face_detection_tflite",    "mp_face");
		maybe_add(s, "dnn_model_face_blendshapes_tflite",  "mp_blend");
	}

	if (profiles.empty()) {
		profiles = {
			"fast", "sface", "lresnet100", "openface",
			"yunet", "emotion", "keypoints",
			"det_uint8", "det_caffe", "det_fp16",
			"mp_landmark", "mp_face", "mp_blend"
		};
	} else {
		std::sort(profiles.begin(), profiles.end());
		profiles.erase(std::unique(profiles.begin(), profiles.end()),
					   profiles.end());
	}

	return profiles;
}

// ---------------------- HELP TRAINING ----------------------

static void fa_usage_training(const char *prog, const std::string &config_path)
{
	std::cerr <<
	"Usage: " << prog << " [options]\n"
	"  -u, --user USER             User name\n"
	"  -m, --method METHOD         lbph|eigen|fisher|dnn [default: lbph]\n"
	"  -i, --input-dir DIR         Training images dir\n"
	"  -o, --output-model FILE     Output model XML\n"
	"  -c, --config FILE           Config file (default " FACIALAUTH_CONFIG_DEFAULT ")\n"
	"  -f, --force                 Force overwrite\n"
	"      --dnn-type T            caffe|tensorflow|onnx|openvino|tflite|torch\n"
	"      --dnn-model PATH        DNN model path\n"
	"      --dnn-proto PATH        DNN proto/config path\n"
	"      --dnn-device DEV        cpu|cuda|opencl|openvino\n"
	"      --dnn-threshold VAL     DNN threshold [0-1], default 0.6\n";

	auto profiles = fa_get_dnn_profiles_from_config(config_path);
	std::cerr << "      --dnn-profile NAME      DNN profiles available in config:\n";
	if (profiles.empty()) {
		std::cerr << "                              (none)\n";
	} else {
		for (const auto &p : profiles) {
			std::cerr << "                              " << p << "\n";
		}
	}

	std::cerr <<
	"      --debug                 Enable debug logging\n"
	"      --help                  Show this help\n";
}

// ---------------------- HELP CAPTURE -----------------------

static void fa_usage_capture(const char *prog)
{
	std::cout <<
	"Usage: " << prog << " [OPTIONS]\n"
	"Options:\n"
	"  -u, --user <name>            User name (REQUIRED)\n"
	"  -d, --device <dev>           Video device (/dev/video0)\n"
	"  -w, --width <px>             Frame width\n"
	"  -h, --height <px>            Frame height\n"
	"  -n, --frames <num>           Number of frames\n"
	"  -c, --config <file>          Config file path (default " FACIALAUTH_CONFIG_DEFAULT ")\n"
	"  -f, --force                  Overwrite images (reset index to 1)\n"
	"      --clean                  Delete ALL images for user\n"
	"      --reset                  Delete images + model\n"
	"      --list                   List user images\n"
	"      --format <ext>           Image format (png,jpg) [default: jpg]\n"
	"      --detector-profile NAME  DNN detector profile (det_uint8|det_caffe|det_fp16|yunet)\n"
	"      --debug                  Enable verbose debug\n"
	"      --help                   Show this help\n";
}

// ---------------------- HELP TEST --------------------------

static void fa_usage_test(const char *prog)
{
	std::cerr <<
	"Usage: " << prog << " [options]\n"
	"  -u, --user USER             User name\n"
	"  -m, --model FILE            Model XML (default: basedir/models/<user>.xml)\n"
	"  -c, --config FILE           Config file (default " FACIALAUTH_CONFIG_DEFAULT ")\n"
	"      --frames N              Number of frames\n"
	"      --debug                 Enable debug logging\n"
	"      --help                  Show this help\n";
}

// ======================================================================
// CLI IMPLEMENTATIONS
// ======================================================================

// ---------------------- TRAINING CLI ----------------------------------

int fa_training_cli(int argc, char *argv[])
{
	FacialAuthConfig cfg;

	std::string user;
	std::string input_dir;
	std::string method = "lbph";
	std::string log;
	std::string output_model;
	std::string config_path = FACIALAUTH_CONFIG_DEFAULT;
	std::string dnn_profile_cli;

	bool force     = false;
	bool debug_cli = false;

	static struct option long_opts[] = {
		{"user",          required_argument, nullptr, 'u'},
		{"method",        required_argument, nullptr, 'm'},
		{"input-dir",     required_argument, nullptr, 'i'},
		{"output-model",  required_argument, nullptr, 'o'},
		{"config",        required_argument, nullptr, 'c'},
		{"force",         no_argument,       nullptr, 'f'},
		{"dnn-type",      required_argument, nullptr,  1 },
		{"dnn-model",     required_argument, nullptr,  2 },
		{"dnn-proto",     required_argument, nullptr,  3 },
		{"dnn-device",    required_argument, nullptr,  4 },
		{"dnn-threshold", required_argument, nullptr,  5 },
		{"debug",         no_argument,       nullptr,  6 },
		{"dnn-profile",   required_argument, nullptr,  7 },
		{"help",          no_argument,       nullptr,  8 },
		{nullptr,         0,                 nullptr,  0 }
	};

	int opt, idx;
	optind = 1;

	while ((opt = getopt_long(argc, argv, "u:m:i:o:c:f", long_opts, &idx)) != -1) {
		switch (opt) {
			case 'u':
				user = optarg;
				break;
			case 'm':
				method = optarg;
				break;
			case 'i':
				input_dir = optarg;
				break;
			case 'o':
				output_model = optarg;
				break;
			case 'c':
				config_path = optarg;
				break;
			case 'f':
				force = true;
				cfg.force_overwrite = true;
				break;
			case 1:
				cfg.dnn_type = optarg;
				break;
			case 2:
				cfg.dnn_model_path = optarg;
				break;
			case 3:
				cfg.dnn_proto_path = optarg;
				break;
			case 4:
				cfg.dnn_device = optarg;
				break;
			case 5:
				cfg.dnn_threshold = std::stod(optarg);
				break;
			case 6:
				debug_cli = true;
				break;
			case 7:
				dnn_profile_cli = optarg;
				break;
			case 8:
				fa_usage_training(argv[0], config_path);
				return 0;
			default:
				fa_usage_training(argv[0], config_path);
				return 1;
		}
	}

	if (user.empty()) {
		fa_usage_training(argv[0], config_path);
		return 1;
	}

	// Load configuration (if missing, defaults remain)
	fa_load_config(config_path, cfg, log);

	// CLI debug overrides config
	if (debug_cli) {
		cfg.debug = true;
		std::cout << "[DEBUG] Debug mode FORZATO da CLI (--debug)\n";
	}

	// For DNN method, select profile
	{
		std::string mt = to_lower_str(method);
		if (mt == "dnn") {
			std::string profile_to_use =
			!dnn_profile_cli.empty() ? dnn_profile_cli : cfg.dnn_profile;

			if (!profile_to_use.empty()) {
				if (!fa_select_dnn_profile(cfg, profile_to_use, log)) {
					std::cerr << "[ERROR] Unknown DNN profile: "
					<< profile_to_use << "\n";
					if (!log.empty())
						std::cerr << log << "\n";
					return 1;
				}
			}
		}
	}

	if (!fa_train_user(user, cfg, method, input_dir, output_model, force, log)) {
		std::cerr << "[ERROR] Training failed for user " << user << "\n";
		if (!log.empty())
			std::cerr << log << "\n";
		return 1;
	}

	std::cout << "[INFO] Training completed for user " << user << "\n";
	return 0;
}

// ---------------------- CAPTURE CLI ----------------------------------

int fa_capture_cli(int argc, char *argv[])
{
	FacialAuthConfig cfg;

	std::string config_path = FACIALAUTH_CONFIG_DEFAULT;
	std::string user;
	std::string img_format = "jpg";
	std::string log;
	std::string detector_profile_cli;

	bool force       = false;
	bool clean       = false;
	bool reset       = false;
	bool list_images = false;
	bool debug_cli   = false;

	int width  = 0;
	int height = 0;
	int frames = 0;

	static struct option long_opts[] = {
		{"user",             required_argument, nullptr, 'u'},
		{"device",           required_argument, nullptr, 'd'},
		{"width",            required_argument, nullptr, 'w'},
		{"height",           required_argument, nullptr, 'h'},
		{"frames",           required_argument, nullptr, 'n'},
		{"config",           required_argument, nullptr, 'c'},
		{"force",            no_argument,       nullptr, 'f'},
		{"format",           required_argument, nullptr,  1 },
		{"debug",            no_argument,       nullptr,  2 },
		{"clean",            no_argument,       nullptr,  3 },
		{"reset",            no_argument,       nullptr,  4 },
		{"list",             no_argument,       nullptr,  5 },
		{"help",             no_argument,       nullptr,  6 },
		{"detector-profile", required_argument, nullptr,  7 },
		{nullptr,            0,                 nullptr,  0 }
	};

	int opt, idx;
	optind = 1;

	while ((opt = getopt_long(argc, argv, "u:d:w:h:n:c:f", long_opts, &idx)) != -1) {
		switch (opt) {
			case 'u':
				user = optarg;
				break;
			case 'd':
				cfg.device = optarg;
				break;
			case 'w':
				width = std::stoi(optarg);
				break;
			case 'h':
				height = std::stoi(optarg);
				break;
			case 'n':
				frames = std::stoi(optarg);
				break;
			case 'c':
				config_path = optarg;
				break;
			case 'f':
				force = true;
				cfg.force_overwrite = true;
				break;
			case 1:
				img_format = optarg;
				break;
			case 2:
				debug_cli = true;
				break;
			case 3:
				clean = true;
				break;
			case 4:
				reset = true;
				break;
			case 5:
				list_images = true;
				break;
			case 6:
				fa_usage_capture(argv[0]);
				return 0;
			case 7:
				detector_profile_cli = optarg;
				break;
			default:
				fa_usage_capture(argv[0]);
				return 1;
		}
	}

	if (user.empty()) {
		std::cerr << "[ERROR] Missing --user\n";
		fa_usage_capture(argv[0]);
		return 1;
	}

	fa_load_config(config_path, cfg, log);

	if (debug_cli) {
		cfg.debug = true;
		std::cout << "[DEBUG] Debug mode FORZATO da CLI (--debug)\n";
	}

	if (width > 0)
		cfg.width = width;
	if (height > 0)
		cfg.height = height;
	if (frames > 0)
		cfg.frames = frames;

	if (!detector_profile_cli.empty())
		cfg.detector_profile = to_lower_str(detector_profile_cli);

	if (list_images) {
		fa_list_images(cfg, user);
		return 0;
	}

	if (clean) {
		std::cout << "[INFO] Removing all images for user " << user << "\n";
		fa_clean_images(cfg, user);
		return 0;
	}

	if (reset) {
		std::cout << "[INFO] Reset user data (images + model)\n";
		fa_clean_images(cfg, user);
		fa_clean_model(cfg, user);
		return 0;
	}

	if (force) {
		std::cout << "[INFO] FORCE enabled: cleaning images before capture\n";
		fa_clean_images(cfg, user);
	}

	std::cout << "[INFO] Starting capture for user: " << user << "\n";

	if (!fa_capture_images(user, cfg, force, log, img_format)) {
		std::cerr << "[ERROR] Capture failed\n";
		if (!log.empty())
			std::cerr << log << "\n";
		return 1;
	}

	std::cout << "[INFO] Capture completed\n";
	return 0;
}

// ---------------------- TEST CLI -------------------------------------

int fa_test_cli(int argc, char *argv[])
{
	FacialAuthConfig cfg;

	std::string user;
	std::string model_path;
	std::string config_path = FACIALAUTH_CONFIG_DEFAULT;
	std::string log;

	int frames     = 0;
	bool debug_cli = false;

	static struct option long_opts[] = {
		{"user",    required_argument, nullptr, 'u'},
		{"model",   required_argument, nullptr, 'm'},
		{"config",  required_argument, nullptr, 'c'},
		{"frames",  required_argument, nullptr,  1 },
		{"debug",   no_argument,       nullptr,  2 },
		{"help",    no_argument,       nullptr,  3 },
		{nullptr,   0,                 nullptr,  0 }
	};

	int opt, idx;
	optind = 1;

	while ((opt = getopt_long(argc, argv, "u:m:c:", long_opts, &idx)) != -1) {
		switch (opt) {
			case 'u':
				user = optarg;
				break;
			case 'm':
				model_path = optarg;
				break;
			case 'c':
				config_path = optarg;
				break;
			case 1:
				frames = std::stoi(optarg);
				break;
			case 2:
				debug_cli = true;
				break;
			case 3:
				fa_usage_test(argv[0]);
				return 0;
			default:
				fa_usage_test(argv[0]);
				return 1;
		}
	}

	if (user.empty()) {
		fa_usage_test(argv[0]);
		return 1;
	}

	fa_load_config(config_path, cfg, log);

	if (debug_cli) {
		cfg.debug = true;
		std::cout << "[DEBUG] Debug mode FORZATO da CLI (--debug)\n";
	}

	if (frames > 0)
		cfg.frames = frames;

	if (model_path.empty())
		model_path = fa_user_model_path(cfg, user);

	double best_conf  = 0.0;
	int    best_label = -1;

	bool ok = fa_test_user(user, cfg, model_path, best_conf, best_label, log);

	std::cout << "Result: " << (ok ? "SUCCESS" : "FAIL") << "\n"
	<< "  best_conf = " << best_conf << "\n"
	<< "  best_label = " << best_label << "\n";

	if (!log.empty())
		std::cout << log << "\n";

	return ok ? 0 : 1;
}
