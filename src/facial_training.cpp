#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <filesystem>
#include <vector>
#include <string>
#include <unistd.h> // getuid()

namespace fs = std::filesystem;

void trainModel(const std::string &trainingDataDir,
				const std::string &outputFile,
				const std::string &method)
{
	std::vector<cv::Mat> images;
	std::vector<int> labels;

	int label = 0;

	std::cout << "[DEBUG] Scanning directory: " << trainingDataDir << std::endl;

	for (const auto &entry : fs::directory_iterator(trainingDataDir))
	{
		if (entry.is_regular_file())
		{
			std::string ext = entry.path().extension().string();
			if (ext == ".jpg" || ext == ".png" || ext == ".jpeg")
			{
				cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
				if (img.empty())
				{
					std::cerr << "[WARN] Cannot read image: " << entry.path() << std::endl;
					continue;
				}

				std::cout << "[INFO] Found image: " << entry.path() << std::endl;
				images.push_back(img);
				labels.push_back(label++);
			}
		}
	}

	if (images.empty())
	{
		std::cerr << "[ERROR] No training images found in " << trainingDataDir << std::endl;
		return;
	}

	cv::Ptr<cv::face::FaceRecognizer> model;

	if (method == "lbph")
		model = cv::face::LBPHFaceRecognizer::create();
	else if (method == "eigen")
		model = cv::face::EigenFaceRecognizer::create();
	else if (method == "fisher")
		model = cv::face::FisherFaceRecognizer::create();
	else
	{
		std::cerr << "[ERROR] Invalid method. Use: lbph | eigen | fisher" << std::endl;
		return;
	}

	std::cout << "[INFO] Training model with " << images.size() << " images..." << std::endl;
	model->train(images, labels);

	fs::path outputPath(outputFile);
	fs::create_directories(outputPath.parent_path());

	model->save(outputFile);
	std::cout << "[SUCCESS] Model saved to: " << outputFile << std::endl;
}

int main(int argc, char **argv)
{
	// Impedisci esecuzione da utente non root
	if (getuid() != 0)
	{
		std::cerr << "[ERROR] This program must be run as root!" << std::endl;
		return 1;
	}

	if (argc < 5)
	{
		std::cerr << "Usage: facial_training -u <user> -m <method> <training_data_directory> "
		"[-o|--output <output_file>]\n"
		"Methods: lbph, eigen, fisher"
		<< std::endl;
		return 1;
	}

	std::string user;
	std::string method = "lbph";
	std::string trainingDataDir;
	std::string outputFile;

	for (int i = 1; i < argc; ++i)
	{
		std::string arg = argv[i];

		if ((arg == "-u" || arg == "--user") && i + 1 < argc)
			user = argv[++i];
		else if ((arg == "-m" || arg == "--method") && i + 1 < argc)
			method = argv[++i];
		else if ((arg == "-o" || arg == "--output") && i + 1 < argc)
			outputFile = argv[++i];
		else if (arg[0] != '-')
			trainingDataDir = arg;
	}

	if (user.empty())
	{
		std::cerr << "[ERROR] Missing user (-u <user>)" << std::endl;
		return 1;
	}

	if (trainingDataDir.empty())
	{
		std::cerr << "[ERROR] Missing training data directory" << std::endl;
		return 1;
	}

	if (outputFile.empty())
	{
		outputFile = "/etc/pam_facial_auth/" + user + "/models/" + user + ".xml";
		std::cout << "[INFO] No output specified, defaulting to: " << outputFile << std::endl;
	}

	std::cout << "[INFO] User: " << user << std::endl;
	std::cout << "[INFO] Method: " << method << std::endl;
	std::cout << "[INFO] Training data: " << trainingDataDir << std::endl;
	std::cout << "[INFO] Output file: " << outputFile << std::endl;

	trainModel(trainingDataDir, outputFile, method);

	return 0;
}
