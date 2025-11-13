#include "../include/libfacialauth.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdarg>
#include <cerrno>

// ----------------------------------------------------------
// Funzioni di utilità
// ----------------------------------------------------------
std::string trim(const std::string &s) {
	size_t b = s.find_first_not_of(" \t\r\n");
	if (b == std::string::npos) return "";
	size_t e = s.find_last_not_of(" \t\r\n");
	return s.substr(b, e - b + 1);
}

bool str_to_bool(const std::string &s, bool defval) {
	std::string t = trim(s);
	for (auto &c : t) c = ::tolower(c);
	if (t == "1" || t == "true" || t == "yes" || t == "on")  return true;
	if (t == "0" || t == "false" || t == "no"  || t == "off") return false;
	return defval;
}

bool read_kv_config(const std::string &path,
					FacialAuthConfig &cfg,
					std::string *logbuf)
{
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

		if (key == "device") cfg.device = val;
		else if (key == "width") cfg.width = std::max(64, std::stoi(val));
		else if (key == "height") cfg.height = std::max(64, std::stoi(val));
		else if (key == "threshold") cfg.threshold = std::stod(val);
		else if (key == "timeout") cfg.timeout = std::max(1, std::stoi(val));
		else if (key == "nogui") cfg.nogui = str_to_bool(val, cfg.nogui);
		else if (key == "debug") cfg.debug = str_to_bool(val, cfg.debug);
		else if (key == "model") cfg.model = val;
		else if (key == "detector") cfg.detector = val;
		else if (key == "model_format") cfg.model_format = val;
		else if (key == "frames") cfg.frames = std::max(1, std::stoi(val));
		else if (key == "fallback_device")
			cfg.fallback_device = str_to_bool(val, cfg.fallback_device);
		else if (key == "model_path")
			cfg.model_path = val;
	}
	return true;
}

void ensure_dirs(const std::string &path) {
	if (path.empty()) return;

	std::string cur;
	std::stringstream ss(path);
	std::string tok;

	if (path[0] == '/') cur = "/";

	while (std::getline(ss, tok, '/')) {
		if (tok.empty()) continue;
		if (cur.size() > 1) cur += "/";
		cur += tok;
		::mkdir(cur.c_str(), 0755);
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
	::usleep(ms * 1000);
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

bool open_camera(const FacialAuthConfig &cfg,
				 cv::VideoCapture &cap,
				 std::string &device_used)
{
	device_used = cfg.device;
	cap.open(cfg.device);

	if (!cap.isOpened() && cfg.fallback_device) {
		log_tool(cfg.debug, "WARN",
				 "Primary device %s failed, trying /dev/video1",
		   cfg.device.c_str());
		cap.open("/dev/video1");
		if (cap.isOpened())
			device_used = "/dev/video1";
	}

	if (!cap.isOpened())
		return false;

	(void) cap.set(cv::CAP_PROP_FRAME_WIDTH,  cfg.width);
	(void) cap.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.height);
	return true;
}

// ----------------------------------------------------------
// Implementazione FaceRecWrapper
// ----------------------------------------------------------
FaceRecWrapper::FaceRecWrapper(const std::string& basePath_,
							   const std::string& name,
							   const std::string& model_type)
: modelType(model_type), basePath(basePath_)
{
	(void)name; // per ora non usato
	recognizer = cv::face::LBPHFaceRecognizer::create();

	// Carica Haar cascade dalla basedir, oppure da sistema
	std::string haarp = join_path(basePath, "haarcascades/haarcascade_frontalface_default.xml");
	if (!file_exists(haarp)) {
		// fallback su path di sistema
		haarp = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml";
	}

	if (!haarp.empty() && file_exists(haarp)) {
		if (!faceCascade.load(haarp)) {
			std::cerr << "[FA-WARN] Impossibile caricare Haar cascade da: "
			<< haarp << "\n";
		}
	}
}

void FaceRecWrapper::Train(const std::vector<cv::Mat>& images,
						   const std::vector<int>& labels)
{
	recognizer->train(images, labels);
}

void FaceRecWrapper::Recognize(cv::Mat& face) {
	int label = -1;
	double confidence = 0.0;
	recognizer->predict(face, label, confidence);
}

void FaceRecWrapper::Load(const std::string& modelFile) {
	recognizer->read(modelFile);
}

void FaceRecWrapper::Save(const std::string& modelFile) {
	recognizer->write(modelFile);
}

void FaceRecWrapper::Predict(cv::Mat& face,
							 int& prediction,
							 double& confidence)
{
	recognizer->predict(face, prediction, confidence);
}

bool FaceRecWrapper::DetectFace(const cv::Mat& frame, cv::Rect& faceROI) {
	if (frame.empty())
		return false;

	if (faceCascade.empty()) {
		std::cerr << "[FA-WARN] Haar cascade non caricato.\n";
		return false;
	}

	cv::Mat gray;
	cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
	cv::equalizeHist(gray, gray);

	std::vector<cv::Rect> faces;
	faceCascade.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(80, 80));

	if (faces.empty())
		return false;

	// scegli il volto più grande
	int bestArea = 0;
	size_t idx = 0;
	for (size_t i = 0; i < faces.size(); ++i) {
		int area = faces[i].area();
		if (area > bestArea) {
			bestArea = area;
			idx = i;
		}
	}
	faceROI = faces[idx];
	return true;
}

// cattura N immagini del volto e le salva in basePath/<user>/images
bool FaceRecWrapper::CaptureImages(const std::string &user,
								   const FacialAuthConfig &cfg)
{
	// directory destinazione: model_path/user/images
	std::string user_dir = join_path(cfg.model_path, user);
	std::string img_dir  = join_path(user_dir, "images");
	ensure_dirs(img_dir);

	cv::VideoCapture cap;
	std::string dev_used;
	if (!open_camera(cfg, cap, dev_used)) {
		std::cerr << "[FA-ERR] Impossibile aprire la webcam: "
		<< cfg.device << "\n";
		return false;
	}

	std::cout << "[FA-INFO] Cattura immagini utente '" << user
	<< "' da " << dev_used
	<< " (" << cfg.width << "x" << cfg.height << ")\n";

	int saved = 0;
	int attempts = 0;
	const int max_attempts = cfg.frames * 20; // per sicurezza

	while (saved < cfg.frames && attempts < max_attempts) {
		cv::Mat frame;
		cap >> frame;
		if (frame.empty()) {
			std::cerr << "[FA-ERR] Frame vuoto dalla camera.\n";
			break;
		}
		attempts++;

		cv::Rect roi;
		if (!DetectFace(frame, roi)) {
			if (cfg.debug)
				std::cerr << "[FA-DEBUG] Nessun volto rilevato in questo frame.\n";
			continue;
		}

		// estrai volto e converti in grigio
		cv::Mat face = frame(roi).clone();
		cv::Mat gray;
		cv::cvtColor(face, gray, cv::COLOR_BGR2GRAY);

		// opzionale: normalizza dimensione, es. 200x200
		cv::Mat resized;
		cv::resize(gray, resized, cv::Size(200, 200));

		std::string filename = join_path(
			img_dir,
			"img_" + std::to_string(saved + 1) + ".png"
		);

		if (!cv::imwrite(filename, resized)) {
			std::cerr << "[FA-ERR] Impossibile salvare " << filename << "\n";
		} else {
			std::cout << "[FA-INFO] Salvata immagine: " << filename << "\n";
			saved++;
		}

		// pausa tra una cattura e l'altra
		if (cfg.timeout > 0)
			sleep_ms(cfg.timeout * 1000);
	}

	cap.release();

	if (saved == 0) {
		std::cerr << "[FA-ERR] Nessuna immagine salvata.\n";
		return false;
	}
	std::cout << "[FA-INFO] Cattura completata. Immagini salvate: "
	<< saved << "\n";
	return true;
}

// ----------------------------------------------------------
// Implementazione FacialAuth (stub ragionevole)
// ----------------------------------------------------------
FacialAuth::FacialAuth() {
	recognizer = cv::face::LBPHFaceRecognizer::create();

	FacialAuthConfig cfg;
	if (read_kv_config("/etc/security/pam_facial.conf", cfg, nullptr)) {
		modelPath = cfg.model_path;
	} else {
		modelPath = "/etc/pam_facial_auth/models";
	}
}

FacialAuth::~FacialAuth() = default;

bool FacialAuth::LoadModel(const std::string &modelPath_) {
	try {
		recognizer->read(modelPath_);
		modelPath = modelPath_;
		return true;
	} catch (const std::exception &e) {
		std::cerr << "[FA-ERR] Errore caricamento modello: "
		<< e.what() << "\n";
		return false;
	}
}

bool FacialAuth::RecognizeFace(const cv::Mat &faceImage) {
	if (recognizer->empty()) {
		std::cerr << "[FA-ERR] Modello non caricato.\n";
		return false;
	}

	int label = -1;
	double confidence = 0.0;
	recognizer->predict(faceImage, label, confidence);

	if (confidence < 50.0) {
		std::cout << "[FA-INFO] Autenticazione OK, label=" << label
		<< " conf=" << confidence << "\n";
		return true;
	} else {
		std::cout << "[FA-INFO] Autenticazione FALLITA, conf="
		<< confidence << "\n";
		return false;
	}
}

bool FacialAuth::Authenticate(const std::string &user) {
	(void)user;
	// TODO: integrazione completa con PAM: carica modello user, cattura volto, ecc.
	std::cerr << "[FA-INFO] Authenticate() stub.\n";
	return false;
}

bool FacialAuth::TrainModel(const std::vector<cv::Mat> &images,
							const std::vector<int> &labels)
{
	if (images.empty() || labels.empty() || images.size() != labels.size())
		return false;
	recognizer->train(images, labels);
	return true;
}
