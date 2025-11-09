#include "FaceRecWrapper.h"
#include <opencv2/face.hpp>  // For FaceRecognizer

// Constructor
FaceRecWrapper::FaceRecWrapper() : sizeFace(96) {}

FaceRecWrapper::FaceRecWrapper(const std::string &techniqueName, const std::string &pathCascade) : sizeFace(96) {
	SetTechnique(techniqueName);
	LoadCascade(pathCascade);
}

// Training the recognizer
void FaceRecWrapper::Train(const std::vector<cv::Mat> &images, const std::vector<int> &labels) {
	std::vector<cv::Mat> imagesCropped;
	std::vector<int> labelsCropped;

	for (size_t i = 0; i < images.size(); ++i) {
		cv::Mat crop;
		if (CropFace(images[i], crop)) {
			labelsCropped.push_back(labels[i]);
			imagesCropped.push_back(crop);
		}
	}

	fr->train(imagesCropped, labelsCropped);
}

// Predicting labels
void FaceRecWrapper::Predict(const cv::Mat &im, int &label, double &confidence) {
	cv::Mat cropped;
	if (!CropFace(im, cropped)) {
		label = -1;
		confidence = 10000;
		return;
	}

	fr->predict(cropped, label, confidence);
}

// Saving the model
void FaceRecWrapper::Save(const std::string &path) {
	fr->save(path + "-facerec.xml");
}

// Loading the model
void FaceRecWrapper::Load(const std::string &path) {
	fr->load(cv::String(path + "-facerec.xml"));
}

// Setting label names
void FaceRecWrapper::SetLabelNames(const std::vector<std::string> &names) {
	labelNames = names;
}

// Getting label name by index
std::string FaceRecWrapper::GetLabelName(int index) {
	if (index >= 0 && index < labelNames.size()) {
		return labelNames[index];
	}
	return "Unknown";
}

// Set the recognition technique
bool FaceRecWrapper::SetTechnique(const std::string &t) {
	if (t == "eigen") {
		fr = cv::face::EigenFaceRecognizer::create();
	} else if (t == "fisher") {
		fr = cv::face::FisherFaceRecognizer::create();
	} else if (t == "lbph") {
		fr = cv::face::LBPHFaceRecognizer::create();
	} else {
		std::cerr << "Invalid technique: " << t << ", defaulting to eigenfaces." << std::endl;
		fr = cv::face::EigenFaceRecognizer::create();
		technique = "eigen";
		return false;
	}

	technique = t;
	return true;
}

// Load the cascade for face detection
bool FaceRecWrapper::LoadCascade(const std::string &filepath) {
	pathCascade = filepath;
	if (!cascade.load(pathCascade)) {
		std::cerr << "Could not load haar cascade classifier." << std::endl;
		return false;
	}
	return true;
}

// Crop face from the image
bool FaceRecWrapper::CropFace(const cv::Mat &image, cv::Mat &cropped) {
	std::vector<cv::Rect> faces;
	cascade.detectMultiScale(image, faces, 1.05, 8, 0, cv::Size(40, 40));
	if (faces.empty()) {
		return false;
	}

	cropped = image(faces[0]);
	cv::resize(cropped, cropped, cv::Size(sizeFace, sizeFace));
	return true;
}
