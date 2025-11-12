#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include "FaceRecWrapper.h"
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
	std::vector<cv::Mat> images;
	std::vector<int> labels;

	// Percorso per caricare le immagini e le etichette di addestramento
	// Dovresti implementare la logica per caricare immagini e etichette da un dataset
	// per esempio usando un ciclo che carica le immagini da una cartella

	FaceRecWrapper faceRec("path/to/model.xml", "trainer");

	// Usa l'addestramento del riconoscimento facciale
	faceRec.Train(images, labels);

	std::cout << "Addestramento completato!" << std::endl;
	return 0;
}
