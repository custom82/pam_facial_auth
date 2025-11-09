#include "FaceRecWrapper.h"
#include "Utils.h"
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>
#include <vector>

int main(int argc, char ** argv) {
	if (argc < 2) {
		std::cerr << "usage: facial_training <data_dir>" << std::endl;
		return -1;
	}

	std::string pathDir = argv[1];

	// Crea il vettore nullVec prima di passarlo
	std::vector<std::string> nullVec;
	std::vector<std::string> usernames;
	Utils::WalkDirectory(pathDir, nullVec, usernames);

	for (size_t i = 0; i < usernames.size(); ++i) {
		std::vector<std::string> files;
		Utils::WalkDirectory(pathDir + "/" + usernames[i], files, nullVec);

		// Elabora i file qui
		for (size_t j = 0; j < files.size(); ++j) {
			cv::Mat temp = cv::imread(pathDir + "/" + usernames[i] + "/" + files[j], cv::IMREAD_GRAYSCALE);
			if (temp.empty()) {
				std::cerr << "Immagine non trovata o non valida: " << pathDir + "/" + usernames[i] + "/" + files[j] << std::endl;
				continue;
			}
			// Aggiungi immagine e label
		}
	}

	return 0;
}
