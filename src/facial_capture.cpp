#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>  // Assicurati di includere il modulo facciale di OpenCV
#include <iostream>
#include <string>
#include "../include/libfacialauth.h"  // Includi la libreria di riconoscimento facciale

int main(int argc, char **argv) {
    // Verifica se il percorso del modello è stato fornito
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path>" << std::endl;
        return -1;
    }

    // Inizializza l'oggetto FaceRecWrapper con il percorso del modello, il nome e il tipo di modello
    std::string modelPath = argv[1];  // Usa il percorso del modello passato come parametro
    FaceRecWrapper faceRec(modelPath, "user", "LBPH");  // Puoi modificare "user" con un altro nome

    // Acquisizione video dalla webcam
    cv::Mat frame;
    cv::VideoCapture cap(0);  // Utilizza la prima webcam disponibile
    if (!cap.isOpened()) {
        std::cerr << "Could not open camera." << std::endl;
        return -1;
    }

    std::cout << "Press 'q' to quit" << std::endl;

    while (true) {
        cap >> frame;  // Cattura un frame dalla webcam
        if (frame.empty()) {
            std::cerr << "Failed to capture image" << std::endl;
            break;
        }

        // Riconoscimento del volto
        cv::Rect faceROI;  // Area del volto nel frame
        bool faceDetected = false;

        // Verifica la presenza di un volto nel frame
        faceDetected = faceRec.DetectFace(frame, faceROI);  // Metodo per rilevare il volto

        if (faceDetected) {
            cv::rectangle(frame, faceROI, cv::Scalar(0, 255, 0), 2);  // Disegna un rettangolo intorno al volto
            cv::Mat faceRegion = frame(faceROI);  // Estrai la regione del volto

            // Riconosci il volto nella regione
            int label;
            double confidence;
            faceRec.Predict(faceRegion, label, confidence);

            // Visualizza i risultati del riconoscimento
            std::cout << "Predicted label: " << label << ", Confidence: " << confidence << std::endl;

            if (confidence < 50) {
                std::cout << "Face recognized successfully!" << std::endl;
            } else {
                std::cout << "Face recognition failed. Confidence: " << confidence << std::endl;
            }
        }

        // Mostra il frame con il volto rilevato
        cv::imshow("Facial Capture", frame);

        // Interrompe se l'utente preme il tasto 'q'
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    cap.release();  // Rilascia la videocamera
    cv::destroyAllWindows();  // Chiudi tutte le finestre di OpenCV

    return 0;
}
