#include <iostream>
#include <string>
#include <filesystem> // Inclusione necessaria per std::filesystem
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;  // Dichiara il namespace per filesystem

bool test_model(const std::string& model_path) {
	if (!fs::exists(model_path)) {
		std::cerr << "Model not found at " << model_path << std::endl;
		return false;
	}
	std::cout << "Model found at " << model_path << std::endl;
	return true;
}

int main(int argc, char* argv[]) {
	if (argc != 2) {
		std::cerr << "Usage: " << argv[0] << " <model_path>" << std::endl;
		return 1;
	}

	std::string model_path = argv[1];

	if (!test_model(model_path)) {
		return 1;
	}

	std::cout << "Model test passed!" << std::endl;
	return 0;
}
