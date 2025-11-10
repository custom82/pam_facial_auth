#include "FaceRecWrapper.h"
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

int main() {
	cv::Mat image = cv::imread("/path/to/test/image.jpg", cv::IMREAD_GRAYSCALE);
	if (image.empty()) {
		std::cerr << "Immagine non trovata!" << std::endl;
		return -1;
	}

	FaceRecWrapper faceRecWrapper("/path/to/model", "FaceModel");
	faceRecWrapper.Load("/path/to/training_data");

	int prediction;
	double confidence;
	faceRecWrapper.Predict(image, prediction, confidence);

	std::cout << "Predizione: " << prediction << ", Confidenza: " << confidence << std::endl;
	return 0;
}
