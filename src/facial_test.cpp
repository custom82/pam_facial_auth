#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>  // Assicurati di includere il modulo facciale di OpenCV
#include <iostream>
#include <filesystem>  // Per leggere il contenuto di una directory
#include "../include/libfacialauth.h"  // Includi la libreria di riconoscimento facciale

namespace fs = std::filesystem;

int main(int argc, char **argv) {
	// Verifica se il percorso del modello è stato fornito
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " <model_path>" << std::endl;
		return -1;
	}

	std::string modelPath = argv[1];  // Percorso del modello

	// Crea un'istanza di FaceRecWrapper per il riconoscimento (tipo modello: LBPH)
	FaceRecWrapper faceRec(modelPath, "test", "LBPH");

	// Vettori per immagini e etichette
	std::vector<cv::Mat> images;
	std::vector<int> labels;

	// Impostiamo il percorso della cartella contenente le immagini di addestramento
	std::string datasetPath = "dataset";  // Cambia con il percorso della tua cartella di immagini
	int label = 0;  // Etichetta iniziale per l'utente

	// Ciclo per leggere le immagini dalla cartella e aggiungerle ai dati di addestramento
	for (const auto& entry : fs::directory_iterator(datasetPath)) {
		if (entry.is_directory()) {
			// Legge tutte le immagini in una sottocartella (una per utente)
			for (const auto& imgEntry : fs::directory_iterator(entry.path())) {
				if (imgEntry.is_regular_file() && imgEntry.path().extension() == ".jpg") {
					cv::Mat img = cv::imread(imgEntry.path().string(), cv::IMREAD_GRAYSCALE);
					if (!img.empty()) {
						images.push_back(img);  // Aggiungi l'immagine
						labels.push_back(label);  // Aggiungi l'etichetta
					}
				}
			}
			label++;  // Incrementa l'etichetta per il prossimo utente
		}
	}

	if (images.empty()) {
		std::cerr << "No images found in the dataset." << std::endl;
		return -1;
	}

	// Addestra il riconoscitore con le immagini e le etichette
	faceRec.Train(images, labels);  // Assicurati di fornire i dati di addestramento

	// Salva il modello addestrato nel file
	faceRec.Save(modelPath);

	std::cout << "Training complete. Model saved to: " << modelPath << std::endl;

	// Carica il modello salvato per il test
	faceRec.Load(modelPath);

	// Avvia il riconoscimento facciale
	cv::VideoCapture cap(0);  // Usa la prima fotocamera disponibile
	if (!cap.isOpened()) {
		std::cerr << "Could not open camera." << std::endl;
		return -1;
	}

	cv::Mat frame;
	while (cap.read(frame)) {
		if (frame.empty()) {
			std::cerr << "Failed to capture image." << std::endl;
			break;
		}

		// Converti l'immagine in scala di grigi
		cv::Mat gray;
		cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

		// Riconoscimento facciale
		int prediction = -1;
		double confidence = 0.0;
		faceRec.Predict(gray, prediction, confidence);

		// Visualizza il risultato del riconoscimento
		std::string result = "Prediction: " + std::to_string(prediction) + " Confidence: " + std::to_string(confidence);
		cv::putText(frame, result, cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

		// Mostra l'immagine
		cv::imshow("Facial Recognition", frame);

		// Interrompi se l'utente preme 'q'
		if (cv::waitKey(1) == 'q') {
			break;
		}
	}

	cap.release();
	cv::destroyAllWindows();

	return 0;
}
