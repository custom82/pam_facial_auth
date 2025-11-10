#include <iostream>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>  // Assicurati di includere OpenCV se necessario

namespace fs = std::filesystem;

bool test_model(const std::string& model_path) {
	// Verifica se il modello esiste nel percorso specificato
	if (!fs::exists(model_path)) {
		std::cerr << "Model not found at " << model_path << std::endl;
		return false;
	}
	// Aggiungi qui la logica per caricare e testare il modello
	std::cout << "Model found at " << model_path << std::endl;
	return true;
}

int main(int argc, char* argv[]) {
	// Verifica la sintassi della riga di comando
	if (argc != 2) {
		std::cerr << "Usage: " << argv[0] << " <model_path>" << std::endl;
		return 1;
	}

	std::string model_path = argv[1];

	// Esegui il test del modello
	if (!test_model(model_path)) {
		return 1;
	}

	std::cout << "Model test passed!" << std::endl;
	return 0;
}
