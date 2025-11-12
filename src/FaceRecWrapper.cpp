#include "FaceRecWrapper.h"
#include <filesystem>

namespace fs = std::filesystem;

FaceRecWrapper::FaceRecWrapper(const std::string &modelPath_, const std::string &name_, const std::string &model_type)
: modelPath(modelPath_), name(name_), type(model_type)
{
	// Crea recognizer
	if (type == "eigen") {
		fr = cv::face::EigenFaceRecognizer::create();
	} else if (type == "fisher") {
		fr = cv::face::FisherFaceRecognizer::create();
	} else {
		fr = cv::face::LBPHFaceRecognizer::create(); // default
	}
}

void FaceRecWrapper::Load(const std::string &path) {
	if (fs::exists(path)) {
		fr->read(path);
	}
}

int FaceRecWrapper::Predict(const cv::Mat &image, int &prediction, double &confidence) {
	if (image.empty()) return -1;
	cv::Mat gray;
	if (image.channels() == 3) cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
	else gray = image;
	// LBPH/Eigen/Fisher: predict(...) void -> out params
	fr->predict(gray, prediction, confidence);
	return 0;
}

bool FaceRecWrapper::SaveAll(const std::string &basePath, bool save_xml, bool save_yaml) {
	try {
		if (save_xml) fr->write(basePath + ".xml");
		if (save_yaml) fr->write(basePath + ".yaml");
		return true;
	} catch (...) { return false; }
}

