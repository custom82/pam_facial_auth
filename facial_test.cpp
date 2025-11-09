#include "FaceRecWrapper.h"
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>

int main(int argc, char** argv) {
	if (argc < 2) {
		std::cerr << "usage: facial_test <image_path>" << std::endl;
		return -1;
	}

	std::string imagePath = argv[1];
	cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);

	if (image.empty()) {
		std::cerr << "Could not open or find the image!" << std::endl;
		return -1;
	}

	// Carica il modello
	FaceRecWrapper frw("eigen", "etc/haarcascade_frontalface_default.xml");
	frw.Load("model");

	// Esegui la previsione
	int label;
	double confidence;
	frw.Predict(image, label, confidence);

	std::cout << "Predicted label: " << label << " with confidence: " << confidence << std::endl;

	return 0;
}
