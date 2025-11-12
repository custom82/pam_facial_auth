#include "../include/libfacialauth.h"
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>  // OpenCV face module
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <iostream>

// Implementazione della classe FaceRecWrapper

FaceRecWrapper::FaceRecWrapper(const std::string& modelPath, const std::string& name, const std::string& model_type)
: modelType(model_type) {
	recognizer = cv::face::LBPHFaceRecognizer::create();  // Inizializzazione del riconoscitore (assicurarsi che il modulo contrib di OpenCV sia disponibile)
}

void FaceRecWrapper::Train(const std::vector<cv::Mat>& images, const std::vector<int>& labels) {
	recognizer->train(images, labels);
}

void FaceRecWrapper::Recognize(cv::Mat& face) {
	int label = -1;
	double confidence = 0.0;
	recognizer->predict(face, label, confidence);
}

void FaceRecWrapper::Load(const std::string& modelFile) {
	recognizer->read(modelFile);  // Carica il modello dal file
}

void FaceRecWrapper::Save(const std::string& modelFile) {
	recognizer->write(modelFile);  // Salva il modello in un file
}

void FaceRecWrapper::Predict(cv::Mat& face, int& prediction, double& confidence) {
	recognizer->predict(face, prediction, confidence);
}


// Implementazione della classe FacialAuth

FacialAuth::FacialAuth() {
	recognizer = cv::face::LBPHFaceRecognizer::create();  // Inizializzazione del riconoscitore facciale (LBPH)
}

FacialAuth::~FacialAuth() {
	// Distruttore: puoi rilasciare risorse se necessario
}

bool FacialAuth::Authenticate(const std::string &user) {
	// Inizializzazione della variabile per l'immagine del volto
	cv::Mat faceImage;

	// Simuliamo il processo di acquisizione del volto (puoi sostituirlo con l'input dalla webcam o immagine)
	faceImage = cv::imread("path_to_user_image.jpg", cv::IMREAD_GRAYSCALE);

	if (faceImage.empty()) {
		std::cerr << "Errore: Immagine del volto non trovata!" << std::endl;
		return false;
	}

	// Carica il modello se non è già stato caricato
	if (!recognizer->isTrained()) {
		if (!LoadModel("path_to_model.xml")) {
			std::cerr << "Errore: Impossibile caricare il modello facciale!" << std::endl;
			return false;
		}
	}

	// Riconosciamo il volto
	return RecognizeFace(faceImage);
}

bool FacialAuth::LoadModel(const std::string &modelPath) {
	// Carica il modello di riconoscimento facciale
	try {
		recognizer->read(modelPath);
		this->modelPath = modelPath;
		return true;
	} catch (const std::exception &e) {
		std::cerr << "Errore durante il caricamento del modello: " << e.what() << std::endl;
		return false;
	}
}

bool FacialAuth::RecognizeFace(const cv::Mat &faceImage) {
	// Verifica se il volto è riconosciuto
	int label = -1;
	double confidence = 0.0;

	recognizer->predict(faceImage, label, confidence);

	if (confidence < 50) {  // Soglia di confidenza (puoi cambiarla in base ai tuoi test)
		std::cout << "Autenticazione riuscita! Utente: " << label << std::endl;
		return true;
	} else {
		std::cout << "Autenticazione fallita. Confidenza: " << confidence << std::endl;
		return false;
	}
}

// Funzioni di utilità

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
		if (key == "device") cfg.device = val;
		else if (key == "width") cfg.width = std::max(64, std::stoi(val));
		else if (key == "height") cfg.height = std::max(64, std::stoi(val));
		else if (key == "threshold") cfg.threshold = std::stod(val);
		else if (key == "timeout") cfg.timeout = std::max(1, std::stoi(val));
		else if (key == "nogui") cfg.nogui = str_to_bool(val, cfg.nogui);
		else if (key == "debug") cfg.debug = str_to_bool(val, cfg.debug);
		else if (key == "model") cfg.model = val;                   // lbph/eigen/fisher
		else if (key == "detector") cfg.detector = val;             // auto/haar/dnn
		else if (key == "model_format") cfg.model_format = val;     // xml/yaml/onnx/both
		else if (key == "frames") cfg.frames = std::max(1, std::stoi(val));
		else if (key == "fallback_device") cfg.fallback_device = str_to_bool(val, cfg.fallback_device);
		else if (key == "model_path") cfg.model_path = val;
	}
	return true;
}

void ensure_dirs(const std::string &path) {
	std::string p = path;
	if (p.empty()) return;
	std::string cur;
	std::stringstream ss(p);
	std::string tok;
	if (p[0] == '/') cur = "/";
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

void sleep_ms(int ms) { ::usleep(ms * 1000); }

