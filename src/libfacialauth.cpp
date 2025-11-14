#include "../include/libfacialauth.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <filesystem>
#include <iostream>

using namespace std;
namespace fs = std::filesystem;

// -------------------------------------------------------------
// CARICAMENTO RIVELATORI
// -------------------------------------------------------------
bool load_detectors(
	const FacialAuthConfig &cfg,
	cv::CascadeClassifier &haar,
	cv::dnn::Net &dnn,
	bool &use_dnn,
	string &log
) {
	log = "";
	use_dnn = false;

	// ----------- HAAR ------------
	if (!cfg.haar_path.empty()) {
		if (haar.load(cfg.haar_path)) {
			log += "[INFO] Haar caricato: " + cfg.haar_path + "\n";
			return true;
		}
	}

	// ----------- DNN -------------
	if (!cfg.dnn_proto.empty() && !cfg.dnn_model.empty()) {
		try {
			dnn = cv::dnn::readNetFromCaffe(cfg.dnn_proto, cfg.dnn_model);
			use_dnn = true;
			log += "[INFO] DNN caricato.\n";
			return true;
		} catch (...) {
			log += "[ERRORE] impossibile caricare DNN.\n";
		}
	}

	return false;
}

// -------------------------------------------------------------
// RILEVAZIONE VOLTO (HAAR o DNN)
// -------------------------------------------------------------
bool detect_face(
	const FacialAuthConfig &cfg,
	const cv::Mat &frame,
	cv::Rect &face_roi,
	cv::CascadeClassifier &haar,
	cv::dnn::Net &dnn
) {
	// ------------------------------
	// USA DNN
	// ------------------------------
	if (!dnn.empty()) {
		cv::Mat blob = cv::dnn::blobFromImage(
			frame, 1.0, cv::Size(300, 300),
											  cv::Scalar(104.0, 177.0, 123.0), false, false
		);

		dnn.setInput(blob);

		cv::Mat out = dnn.forward();  // shape: 1x1xNx7
		cv::Mat det = out.reshape(1, out.total() / 7);

		for (int i = 0; i < det.rows; i++) {
			float confidence = det.at<float>(i, 2);
			if (confidence < 0.6)
				continue;

			int x1 = det.at<float>(i, 3) * frame.cols;
			int y1 = det.at<float>(i, 4) * frame.rows;
			int x2 = det.at<float>(i, 5) * frame.cols;
			int y2 = det.at<float>(i, 6) * frame.rows;

			face_roi = cv::Rect(
				max(0, x1), max(0, y1),
								min(frame.cols - x1, x2 - x1),
								min(frame.rows - y1, y2 - y1)
			);

			return true;
		}
	}

	// ------------------------------
	// USA HAAR
	// ------------------------------
	vector<cv::Rect> faces;
	haar.detectMultiScale(frame, faces, 1.2, 3);

	if (faces.empty())
		return false;

	face_roi = faces[0];
	return true;
}

// -------------------------------------------------------------
// DetectFace wrapper interno
// -------------------------------------------------------------
bool FaceRecWrapper::DetectFace(const cv::Mat &frame, cv::Rect &faceROI) {
	static cv::CascadeClassifier haar;
	static bool loaded = false;

	if (!loaded) {
		haar.load("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml");
		loaded = true;
	}

	vector<cv::Rect> faces;
	haar.detectMultiScale(frame, faces, 1.2, 3);

	if (faces.empty())
		return false;

	faceROI = faces[0];
	return true;
}

// -------------------------------------------------------------
// CATTURA IMMAGINI
// -------------------------------------------------------------
bool FaceRecWrapper::CaptureImages(const string &user, const FacialAuthConfig &cfg) {
	string dir = "/etc/pam_facial_auth/" + user + "/images";
	fs::create_directories(dir);

	cv::VideoCapture cap(0);
	if (!cap.isOpened())
		return false;

	cap.set(cv::CAP_PROP_FRAME_WIDTH, cfg.frame_width);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.frame_height);

	int count = 1;

	while (count <= 20) {
		cv::Mat frame;
		cap >> frame;
		if (frame.empty())
			continue;

		cv::Rect roi;
		if (!DetectFace(frame, roi))
			continue;

		cv::Mat crop = frame(roi);

		string out = dir + "/img_" + to_string(count) + ".png";
		cv::imwrite(out, crop);

		if (cfg.debug)
			cout << "Salvata: " << out << endl;

		if (cfg.sleep_ms > 0)
			cv::waitKey(cfg.sleep_ms);

		count++;
	}

	return true;
}
