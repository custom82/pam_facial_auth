#include "FaceRecWrapper.h"

FaceRecWrapper::FaceRecWrapper(const std::string& modelPath, const std::string& name, const std::string& model_type)
: modelType(model_type) {
	recognizer = cv::face::LBPHFaceRecognizer::create();  // Initialize the recognizer (ensure OpenCV contrib is available)
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
	recognizer->read(modelFile);  // Load the model from the file
}

void FaceRecWrapper::Predict(cv::Mat& face, int& prediction, double& confidence) {
	recognizer->predict(face, prediction, confidence);
}
