#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>
#include <string>

bool test_model(const std::string& model_path, const std::string& test_image_path) {
	// Verifica se il file del modello esiste
	if (!fs::exists(model_path)) {
		std::cerr << "Model not found!" << std::endl;
		return false;
	}

	// Carica il modello (ad esempio, LBPH)
	cv::Ptr<cv::face::LBPHFaceRecognizer> model = cv::face::LBPHFaceRecognizer::create();
	model->read(model_path);

	// Carica l'immagine di test
	cv::Mat test_image = cv::imread(test_image_path, cv::IMREAD_GRAYSCALE);
	if (test_image.empty()) {
		std::cerr << "Failed to load test image" << std::endl;
		return false;
	}

	// Esegui il riconoscimento
	int label = -1;
	double confidence = 0.0;
	model->predict(test_image, label, confidence);

	// Output dei risultati
	std::cout << "Predicted label: " << label << ", Confidence: " << confidence << std::endl;
	return true;
}

int main(int argc, char **argv) {
	if (argc != 3) {
		std::cerr << "Usage: " << argv[0] << " <model_path> <test_image_path>" << std::endl;
		return -1;
	}

	std::string model_path = argv[1];
	std::string test_image_path = argv[2];

	if (!test_model(model_path, test_image_path)) {
		return -1;
	}

	return 0;
}
