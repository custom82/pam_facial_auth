#include "FaceRecWrapper.h"
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>  // For face recognition

// Assuming Utils is a custom utility class, it needs to be defined or included
#include "Utils.h"

FaceRecWrapper::FaceRecWrapper() :
sizeFace(96) {}

FaceRecWrapper::FaceRecWrapper(const std::string &techniqueName, const std::string &pathCascade) :
sizeFace(96) {
	SetTechnique(techniqueName);
	LoadCascade(pathCascade);
}

void FaceRecWrapper::Train(const std::vector<cv::Mat> &images, const std::vector<int> &labels) {
	if (!images.size()) {
		std::cerr << "Empty vector of training images" << std::endl;
		return;
	}

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

void FaceRecWrapper::Predict(const cv::Mat &im, int &label, double &confidence) {
	cv::Mat cropped;
	if (!CropFace(im, cropped)) {
		label = -1;
		confidence = 10000;
		return;
	}

	fr->predict(cropped, label, confidence);
}

void FaceRecWrapper::Save(const std::string &path) {
	std::ifstream orig(pathCascade, std::ios::binary);
	if (!orig) {
		std::cerr << "Error opening file: " << pathCascade << std::endl;
		return;
	}

	std::ofstream cpy(path + "-cascade.xml", std::ios::binary);
	cpy << orig.rdbuf();

	fr->save(path + "-facerec.xml");

	// Write additional information to the model file
	FILE *pModel;
	pModel = fopen(path.c_str(), "w");
	fprintf(pModel, "technique=%s\n", technique.c_str());
	fprintf(pModel, "sizeFace=%d\n", (int)sizeFace);
	fclose(pModel);
}

void FaceRecWrapper::Load(const std::string &path) {
	std::map<std::string, std::string> model;
	Utils::GetConfig(path, model);

	sizeFace = std::stoi(model["sizeFace"]);
	SetTechnique(model["technique"]);

	LoadCascade(path + "-cascade.xml");
	fr->load(cv::String(path + "-facerec.xml"));  // Cast std::string to cv::String
}

void FaceRecWrapper::SetLabelNames(const std::vector<std::string> &names) {
	// Assuming the FaceRecognizer is capable of setting label names like this.
	// If not, you may need to implement this functionality in your own way.
	labelNames = names;
}

std::string FaceRecWrapper::GetLabelName(int index) {
	if (index >= 0 && index < labelNames.size()) {
		return labelNames[index];
	}
	return "Unknown";
}

bool FaceRecWrapper::SetTechnique(const std::string &t) {
	if (t == "eigen") {
		fr = cv::face::EigenFaceRecognizer::create(10);
	} else if (t == "fisher") {
		fr = cv::face::FisherFaceRecognizer::create();
	} else if (t == "lbph") {
		fr = cv::face::LBPHFaceRecognizer::create();
	} else {
		std::cerr << "Invalid technique: " << t << ", defaulting to eigenfaces." << std::endl;
		fr = cv::face::EigenFaceRecognizer::create(10);
		technique = "eigen";
		return false;
	}

	technique = t;
	return true;
}

bool FaceRecWrapper::LoadCascade(const std::string &filepath) {
	pathCascade = filepath;

	if (!cascade.load(pathCascade)) {
		std::cerr << "Could not load haar cascade classifier." << std::endl;
		return false;
	}
	return true;
}

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
