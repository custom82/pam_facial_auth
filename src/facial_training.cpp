#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <filesystem>
#include <vector>
#include <string>

namespace fs = std::filesystem;

void trainModel(const std::string &trainingDataDir, const std::string &outputFile, const std::string &method) {
	std::vector<cv::Mat> images;
	std::vector<int> labels;

	// Loop through all the images and add them to vectors
	int label = 0;
	for (const auto &entry : fs::directory_iterator(trainingDataDir)) {
		if (entry.is_regular_file() && entry.path().extension() == ".jpg") {
			cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
			if (img.empty()) {
				std::cerr << "Failed to read image: " << entry.path() << std::endl;
				continue;
			}

			images.push_back(img);
			labels.push_back(label);
			label++;
		}
	}

	if (images.empty()) {
		std::cerr << "No training images found!" << std::endl;
		return;
	}

	cv::Ptr<cv::face::FaceRecognizer> model;

	if (method == "lbph") {
		model = cv::face::LBPHFaceRecognizer::create();
	} else if (method == "eigen") {
		model = cv::face::EigenFaceRecognizer::create();
	} else if (method == "fisher") {
		model = cv::face::FisherFaceRecognizer::create();
	} else {
		std::cerr << "Invalid method specified!" << std::endl;
		return;
	}

	// Train the model
	model->train(images, labels);

	// Save the trained model
	model->save(outputFile);
	std::cout << "Model saved to " << outputFile << std::endl;
}

int main(int argc, char **argv) {
	if (argc < 5) {
		std::cerr << "Usage: facial_training -u <user> -m <method> <training_data_directory> [--output <output_file>]" << std::endl;
		return 1;
	}

	std::string user;
	std::string method = "lbph"; // Default method is LBPH
	std::string trainingDataDir;
	std::string outputFile;

	// Parse command-line arguments
	for (int i = 1; i < argc; ++i) {
		std::string arg = argv[i];

		if (arg == "-u" || arg == "--user") {
			user = argv[++i];
		} else if (arg == "-m" || arg == "--method") {
			method = argv[++i];
		} else if (arg == "--output") {
			outputFile = argv[++i];
		} else {
			trainingDataDir = arg; // Assume this is the directory with images
		}
	}

	if (user.empty()) {
		std::cerr << "User not specified!" << std::endl;
		return 1;
	}

	if (trainingDataDir.empty()) {
		std::cerr << "Training data directory not specified!" << std::endl;
		return 1;
	}

	if (outputFile.empty()) {
		std::cerr << "Output file path not specified!" << std::endl;
		return 1;
	}

	std::cout << "Training with method: " << method << " for user: " << user << std::endl;
	std::cout << "Using training data from: " << trainingDataDir << std::endl;
	std::cout << "Model will be saved in: " << outputFile << std::endl;

	trainModel(trainingDataDir, outputFile, method);

	return 0;
}
