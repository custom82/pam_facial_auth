#include "FacialAuth.h"
#include <security/pam_ext.h>
#include <syslog.h>
#include <filesystem>

namespace fs = std::filesystem;

static void pamlog(pam_handle_t *pamh, int prio, const char *fmt, ...) {
	if (!pamh) return;
	char buf[1024];
	va_list ap;
	va_start(ap, fmt);
	vsnprintf(buf, sizeof(buf), fmt, ap);
	va_end(ap);
	pam_syslog(pamh, prio, "%s", buf);
}

static void apply_argv_overrides(FacialAuthConfig &cfg, const char **argv, int argc, std::string *trace) {
	for (int i=0;i<argc;i++) {
		std::string a(argv[i]);
		if (trace) *trace += "argv: " + a + "\n";
		if (a=="debug") cfg.debug = true;
		else if (a=="nogui") cfg.nogui = true;
		else if (a.rfind("threshold=",0)==0) cfg.threshold = std::stod(a.substr(10));
		else if (a.rfind("timeout=",0)==0) cfg.timeout = std::stoi(a.substr(8));
		else if (a.rfind("device=",0)==0) cfg.device = a.substr(7);
		else if (a.rfind("width=",0)==0) cfg.width = std::stoi(a.substr(6));
		else if (a.rfind("height=",0)==0) cfg.height = std::stoi(a.substr(7));
		else if (a.rfind("model=",0)==0) cfg.model = a.substr(6);
		else if (a.rfind("detector=",0)==0) cfg.detector = a.substr(9);
		else if (a.rfind("model_format=",0)==0) cfg.model_format = a.substr(13);
		else if (a.rfind("frames=",0)==0) cfg.frames = std::stoi(a.substr(7));
		else if (a=="fallback_device=0" || a=="fallback_device=false") cfg.fallback_device=false;
		else if (a=="fallback_device=1" || a=="fallback_device=true") cfg.fallback_device=true;
		else if (a.rfind("model_path=",0)==0) cfg.model_path = a.substr(11);
	}
}

void FacialAuth::load_config(FacialAuthConfig &cfg, const char **argv, int argc, std::string *trace) {
	std::string log;
	read_kv_config(DEFAULT_CONF, cfg, &log);
	if (trace) *trace += log;
	apply_argv_overrides(cfg, argv, argc, trace);
}

bool FacialAuth::resolve_user_model(const FacialAuthConfig &cfg, const std::string &user,
									std::string &model_base, std::string &model_file)
{
	std::string user_dir = join_path(cfg.model_path, user);
	ensure_dirs(user_dir);
	std::string models_dir = join_path(user_dir, "models");
	ensure_dirs(models_dir);

	model_base = join_path(models_dir, user);
	// prova xml poi yaml
	std::string xml = model_base + ".xml";
	std::string yaml = model_base + ".yaml";

	if (file_exists(xml)) { model_file = xml; return true; }
	if (file_exists(yaml)) { model_file = yaml; return true; }
	model_file.clear();
	return false;
}

bool FacialAuth::recognize_loop(const FacialAuthConfig &cfg, const std::string &user,
								bool verbose_to_pam, pam_handle_t *pamh,
								double &out_conf)
{
	std::string dev_used;
	cv::VideoCapture cap;
	if (!open_camera(cfg, cap, dev_used)) {
		if (verbose_to_pam) pamlog(pamh, LOG_ERR, "Cannot open camera: %s", cfg.device.c_str());
		else log_tool(cfg.debug, "ERR", "Cannot open camera: %s", cfg.device.c_str());
		return false;
	}

	if (verbose_to_pam) {
		pamlog(pamh, LOG_INFO, "Camera opened: %s (%dx%d)", dev_used.c_str(), cfg.width, cfg.height);
	} else {
		log_tool(cfg.debug, "INFO", "Camera opened: %s (%dx%d)", dev_used.c_str(), cfg.width, cfg.height);
	}

	// Load detectors
	cv::CascadeClassifier haar;
	cv::dnn::Net dnn;
	bool use_dnn=false;
	std::string detlog;
	load_detectors(cfg, haar, dnn, use_dnn, detlog);
	if (verbose_to_pam) pamlog(pamh, LOG_DEBUG, "Detector: %s", detlog.c_str());
	else log_tool(cfg.debug, "DEBUG", "Detector: %s", detlog.c_str());

	// Model
	std::string model_base, model_file;
	if (!resolve_user_model(cfg, user, model_base, model_file)) {
		if (verbose_to_pam) pamlog(pamh, LOG_WARNING, "No model found for user '%s'", user.c_str());
		else log_tool(cfg.debug, "WARN", "No model for %s", user.c_str());
		return false;
	}

	FaceRecWrapper fr(model_file, user, cfg.model);
	fr.Load(model_file);

	auto start = std::chrono::steady_clock::now();
	bool recognized=false;
	out_conf=0.0;

	while (true) {
		cv::Mat frame;
		cap >> frame;
		if (frame.empty()) {
			if (verbose_to_pam) pamlog(pamh, LOG_ERR, "Empty frame");
			else log_tool(cfg.debug, "ERR", "Empty frame");
			continue;
		}

		cv::Rect face;
		if (!detect_face(cfg, frame, face, haar, dnn)) {
			if (cfg.debug) {
				if (verbose_to_pam) pamlog(pamh, LOG_DEBUG, "No face detected");
				else log_tool(cfg.debug, "DEBUG", "No face detected");
			}
			if (!cfg.nogui) {
				cv::imshow("facial_auth", frame);
				if (cv::waitKey(1) == 27) break;
			}
			auto now = std::chrono::steady_clock::now();
			if (std::chrono::duration_cast<std::chrono::seconds>(now-start).count() > cfg.timeout) break;
			continue;
		}

		cv::Mat face_img = frame(face).clone();
		int pred=-1; double conf=0.0;
		fr.Predict(face_img, pred, conf);

		if (cfg.debug) {
			if (verbose_to_pam) pamlog(pamh, LOG_INFO, "Predict=%d conf=%.2f", pred, conf);
			else log_tool(cfg.debug, "INFO", "Predict=%d conf=%.2f", pred, conf);
		}

		if (!cfg.nogui) {
			cv::rectangle(frame, face, cv::Scalar(0,255,0), 2);
			cv::putText(frame, "pred="+std::to_string(pred)+" conf="+std::to_string(conf),
						face.tl()+cv::Point(0,-10), cv::FONT_HERSHEY_SIMPLEX, 0.6, {0,255,0}, 2);
			cv::imshow("facial_auth", frame);
			if (cv::waitKey(1)==27) break;
		}

		if (pred == 1 && conf >= cfg.threshold) {
			recognized = true;
			out_conf = conf;
			break;
		}

		auto now = std::chrono::steady_clock::now();
		if (std::chrono::duration_cast<std::chrono::seconds>(now-start).count() > cfg.timeout) break;
		sleep_ms(150);
	}

	cap.release();
	if (!cfg.nogui) cv::destroyAllWindows();
	return recognized;
}

bool FacialAuth::auto_train_from_camera(const FacialAuthConfig &cfg, const std::string &user) {
	std::string dev_used;
	cv::VideoCapture cap;
	if (!open_camera(cfg, cap, dev_used)) {
		log_tool(cfg.debug, "ERR", "Training: cannot open camera %s", cfg.device.c_str());
		return false;
	}
	cv::CascadeClassifier haar; cv::dnn::Net dnn; bool use_dnn=false; std::string detlog;
	load_detectors(cfg, haar, dnn, use_dnn, detlog);
	log_tool(cfg.debug, "INFO", "Training detector: %s", detlog.c_str());

	std::vector<cv::Mat> images;
	std::vector<int> labels;

	auto start = std::chrono::steady_clock::now();
	while ((int)images.size() < cfg.frames) {
		cv::Mat frame; cap >> frame; if (frame.empty()) continue;
		cv::Rect face;
		if (!detect_face(cfg, frame, face, haar, dnn)) {
			if (!cfg.nogui) { cv::imshow("training", frame); cv::waitKey(1); }
			auto now = std::chrono::steady_clock::now();
			if (std::chrono::duration_cast<std::chrono::seconds>(now-start).count() > cfg.timeout*3) break;
			continue;
		}
		cv::Mat gray; cv::cvtColor(frame(face), gray, cv::COLOR_BGR2GRAY);
		cv::resize(gray, gray, cv::Size(200,200));
		images.push_back(gray); labels.push_back(1);
		if (!cfg.nogui) {
			cv::rectangle(frame, face, {0,255,0}, 2);
			cv::imshow("training", frame); cv::waitKey(1);
		}
		log_tool(true, "DEBUG", "Captured %d/%d", (int)images.size(), cfg.frames);
		sleep_ms(80);
	}
	cap.release(); if (!cfg.nogui) cv::destroyAllWindows();
	if (images.empty()) { log_tool(true, "ERR", "No images captured for training"); return false; }

	// Crea recognizer
	cv::Ptr<cv::face::FaceRecognizer> rec;
	if (cfg.model=="eigen") rec = cv::face::EigenFaceRecognizer::create();
	else if (cfg.model=="fisher") rec = cv::face::FisherFaceRecognizer::create();
	else rec = cv::face::LBPHFaceRecognizer::create();

	rec->train(images, labels);

	// Salva
	std::string model_base, model_file;
	resolve_user_model(cfg, user, model_base, model_file); // crea cartelle
	bool save_xml = (cfg.model_format=="xml" || cfg.model_format=="both");
	bool save_yaml = (cfg.model_format=="yaml" || cfg.model_format=="both");
	try {
		if (save_xml) rec->write(model_base + ".xml");
		if (save_yaml) rec->write(model_base + ".yaml");
	} catch (...) {
		return false;
	}
	return true;
}
