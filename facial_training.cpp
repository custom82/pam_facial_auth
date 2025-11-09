#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "FaceRecWrapper.h"
#include "Utils.h"

int main(int argc, char** argv) {
	if (argc < 2 || argc > 4) {
		std::cerr << "usage: facial_training <data_dir> [<technique>] [--user <username>]" << std::endl;
		return -1;
	}

	std::string pathDir = argv[1];
	std::string technique = "eigen";
	std::string user = "";  // Default: empty means train for all users

	if (argc >= 3) {
		technique = argv[2];
	}

	if (argc == 4 && std::string(argv[3]) == "--user") {
		if (argc != 5) {
			std::cerr << "usage: facial_training <data_dir> [<technique>] --user <username>" << std::endl;
			return -1;
		}
		user = argv[4];
	}

	std::vector<std::string> usernames;
	std::vector<cv::Mat> images;
	std::vector<int> labels;

	// Collect all files in the specified directory
	Utils::WalkDirectory(pathDir, {}, usernames);

	for (size_t i = 0; i < usernames.size(); ++i) {
		// If user is specified, only train for that user
		if (!user.empty() && usernames[i] != user) {
			continue;
		}

		std::vector<std::string> files;
		Utils::WalkDirectory(pathDir + "/" + usernames[i], files, {});

		for (size_t j = 0; j < files.size(); ++j) {
			cv::Mat temp = cv::imread(pathDir + "/" + usernames[i] + "/" + files[j], cv::IMREAD_GRAYSCALE);
			if (temp.data) {
				images.push_back(temp);
				labels.push_back(i);
			}
		}
	}

	if (images.empty()) {
		std::cerr << "No images found for training." << std::endl;
		return -1;
	}

	// Select technique
	FaceRecWrapper frw(technique, "etc/haarcascade_frontalface_default.xml");

	// Do training
	std::cout << "Training " << technique << " model..." << std::endl;
	frw.Train(images, labels);

	// Set usernames
	frw.SetLabelNames(usernames);

	// Write out model
	frw.Save("model");

	// Write default config file
	FILE * pConfig;
	pConfig = fopen("config", "w");
	fprintf(pConfig, "imageCapture=true\n");
	fprintf(pConfig, "imageDir=/var/lib/motioneye/Camera1\n");
	fprintf(pConfig, "timeout=10\n");
	fprintf(pConfig, "threshold=%.2f\n", 1000.0);
	fclose(pConfig);

	std::cout << "Success. Config and model files written." << std::endl;
	return 0;
}
