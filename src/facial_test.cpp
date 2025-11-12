#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include "FaceRecWrapper.h"
#include <iostream>

int main(int argc, char** argv) {
	cv::Mat testImage = cv::imread("path/to/test/image.jpg");
	if (testImage.empty()) {
		std::cerr << "Immagine di test non trovata!" << std::endl;
		return -1;
	}

	FaceRecWrapper faceRec("path/to/model.xml", "test");

	int prediction = -1;
	double confidence = 0;
	faceRec.Predict(testImage, prediction, confidence);

	std::cout << "Predizione: " << prediction << ", Confidenza: " << confidence << std::endl;
	return 0;
}
