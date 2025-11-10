#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <cstdlib>

namespace fs = std::filesystem;

void print_usage() {
	std::cout << "Usage: facial_training -u <user> -m <method> <training_data_directory> [--output|-o <output_file>] [--verbose|-v]\n";
	std::cout << "Methods: lbph, eigen, fisher\n";
}

bool verbose = false;

void log_info(const std::string& msg) {
	if (verbose)
		std::cout << "[INFO] " << msg << std::endl;
}

void log_debug(const std::string& msg) {
	if (verbose)
		std::cout << "[DEBUG] " << msg << std::endl;
}

void log_error(const std::string& msg) {
	std::cerr << "[ERROR] " << msg << std::endl;
}

int main(int argc, char** argv) {
	if (argc < 4) {
		print_usage();
		return 1;
	}

	std::string user;
	std::string method;
	std::string training_data_directory;
	std::string output_model;

	// Parse command line arguments
	for (int i = 1; i < argc; ++i) {
		std::string arg = argv[i];

		if (arg == "-u" || arg == "--user") {
			if (i + 1 < argc)
				user = argv[++i];
		} else if (arg == "-m" || arg == "--method") {
			if (i + 1 < argc)
				method = argv[++i];
		} else if (arg == "--output" || arg == "-o") {
			if (i + 1 < argc)
				output_model = argv[++i];
		} else if (arg == "--verbose" || arg == "-v") {
			verbose = true;
		} else if (arg[0] != '-') {
			training_data_directory = arg;
		}
	}

	if (user.empty() || method.empty() || training_data_directory.empty()) {
		print_usage();
		return 1;
	}

	log_info("User: " + user);
	log_info("Method: " + method);
	log_info("Training data: " + training_data_directory);

	if (output_model.empty()) {
		output_model = training_data_directory + "/models/" + user + "_model.xml";
		log_info("No output specified, using default: " + output_model);
	}

	// Load images
	std::vector<cv::Mat> images;
	std::vector<int> labels;
	int label = 0;

	log_debug("Scanning directory: " + training_data_directory);

	for (const auto& entry : fs::directory_iterator(training_data_directory)) {
		if (entry.is_directory()) continue;

		std::string file_path = entry.path().string();
		std::string ext = entry.path().extension().string();
		if (ext != ".jpg" && ext != ".jpeg" && ext != ".png")
			continue;

		cv::Mat img = cv::imread(file_path, cv::IMREAD_GRAYSCALE);
		if (img.empty()) {
			log_error("Failed to load image: " + file_path);
			continue;
		}

		images.push_back(img);
		labels.push_back(label);

		log_debug("Found image: " + file_path);
	}

	if (images.empty()) {
		log_error("No training images found in " + training_data_directory);
		return 1;
	}

	// Create recognizer
	cv::Ptr<cv::face::FaceRecognizer> model;
	if (method == "lbph") {
		model = cv::face::LBPHFaceRecognizer::create();
	} else if (method == "eigen") {
		model = cv::face::EigenFaceRecognizer::create();
	} else if (method == "fisher") {
		model = cv::face::FisherFaceRecognizer::create();
	} else {
		log_error("Invalid method: " + method);
		return 1;
	}

	log_info("Training model with " + std::to_string(images.size()) + " images...");
	model->train(images, labels);

	fs::path out_path(output_model);
	fs::create_directories(out_path.parent_path());

	log_info("Saving model to: " + output_model);
	model->save(output_model);

	std::cout << "[SUCCESS] Model saved successfully to " << output_model << std::endl;
	return 0;
}
