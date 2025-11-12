#include "FaceRecWrapper.h"
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

FaceRecWrapper::FaceRecWrapper(const std::string& modelPath, const std::string& name, const std::string& model_type)
: modelPath(modelPath), model_type(model_type)
{
	if (model_type == "LBPH") {
		recognizer = cv::face::LBPHFaceRecognizer::create();
	}
	// Add other models as needed
}

void FaceRecWrapper::Recognize(cv::Mat& frame)
{
	// Add face recognition logic
	cv::Mat gray;
	cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

	// Detect face using OpenCV's face detection methods
	// Placeholder logic for face detection
	std::vector<cv::Rect> faces;
	cv::CascadeClassifier faceCascade;
	faceCascade.load("haarcascade_frontalface_default.xml");

	faceCascade.detectMultiScale(gray, faces);

	for (size_t i = 0; i < faces.size(); i++) {
		cv::Mat face = gray(faces[i]);
		int label = -1;
		double confidence = 0.0;
		recognizer->predict(face, label, confidence);

		// Display label and confidence
		cv::rectangle(frame, faces[i], cv::Scalar(255, 0, 0), 2);
		std::string labelText = "ID: " + std::to_string(label);
		cv::putText(frame, labelText, cv::Point(faces[i].x, faces[i].y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(255, 255, 0), 2);
	}
}

void FaceRecWrapper::Train(const std::vector<cv::Mat>& images, const std::vector<int>& labels)
{
	recognizer->train(images, labels);
	recognizer->save(modelPath);  // Save model to file
}
