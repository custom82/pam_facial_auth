#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>  // Ensure the OpenCV face module is included

#include "../include/libfacialauth.h"  // Include the FaceRecWrapper class header

int main(int argc, char **argv) {
	// Check if the model path is provided
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " <model_path>" << std::endl;
		return -1;
	}

	// Create a FaceRecWrapper instance with 3 arguments (model path, name, and model type)
	FaceRecWrapper faceRec("path/to/model.xml", "trainer", "LBPH");

	std::vector<cv::Mat> images;
	std::vector<int> labels;

	// Train the recognizer
	faceRec.Train(images, labels);  // Ensure you provide image data for training

	return 0;
}
