#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>  // Ensure the OpenCV face module is included

#include "../include/FaceRecWrapper.h"  // Include the FaceRecWrapper class header

int main(int argc, char **argv) {
	// Check if the model path is provided
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " <model_path>" << std::endl;
		return -1;
	}

	// Create a FaceRecWrapper instance with 3 arguments (model path, name, and model type)
	FaceRecWrapper faceRec("path/to/model.xml", "test", "LBPH");

	cv::Mat testImage = cv::imread("test_image.jpg", cv::IMREAD_GRAYSCALE);

	int prediction = -1;
	double confidence = 0.0;

	// Predict the result
	faceRec.Predict(testImage, prediction, confidence);

	std::cout << "Prediction: " << prediction << ", Confidence: " << confidence << std::endl;

	return 0;
}
