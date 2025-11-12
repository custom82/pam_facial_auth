#include <iostream>
#include <opencv2/opencv.hpp>
#include "FaceRecWrapper.h"

int main(int argc, char** argv)
{
	// Ensure a model file and username are passed
	if (argc < 3) {
		std::cerr << "Usage: " << argv[0] << " <path_to_model> <username>" << std::endl;
		return -1;
	}

	// Initialize FaceRecWrapper for training
	FaceRecWrapper faceRec(argv[1], argv[2], "LBPH");  // Assuming model type is LBPH

	// Collect training data
	std::vector<cv::Mat> images;
	std::vector<int> labels;

	// Your face image collection code here

	faceRec.Train(images, labels);  // Train with the collected images

	std::cout << "Training completed successfully!" << std::endl;
	return 0;
}
