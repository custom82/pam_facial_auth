#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <filesystem>
#include <string>
#include <unistd.h>  // Per getopt
#include "../include/libfacialauth.h"  // Includi la libreria di riconoscimento facciale

namespace fs = std::filesystem;

// Funzione di stampa per l'help
void print_usage(const char* prog_name) {
    std::cerr << "Usage: " << prog_name << " -u <user> [options]" << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  -u, --user <name>       Nome utente per cui salvare le immagini" << std::endl;
    std::cerr << "  -c, --config <file>     File di configurazione (default: /etc/pam_facial_auth/pam_facial.conf)" << std::endl;
    std::cerr << "  -d, --device <path>     Device della webcam (es: /dev/video0)" << std::endl;
    std::cerr << "  -w, --width <px>        Larghezza frame" << std::endl;
    std::cerr << "  -h, --height <px>       Altezza frame" << std::endl;
    std::cerr << "  -f, --force             Sovrascrive immagini esistenti e riparte da 1" << std::endl;
    std::cerr << "  --flush, --clean        Elimina tutte le immagini per l'utente specificato" << std::endl;
    std::cerr << "  -n, --num_images <num>  Numero di immagini da acquisire" << std::endl;
    std::cerr << "  -s, --sleep <sec>       Pausa tra una cattura e l'altra (in secondi)" << std::endl;
    std::cerr << "  -v, --verbose           Output dettagliato" << std::endl;
    std::cerr << "  --debug                 Abilita output di debug" << std::endl;
    std::cerr << "  --nogui                 Disabilita GUI, cattura solo da console" << std::endl;
    std::cerr << "  --help, -H              Mostra questo messaggio" << std::endl;
}

// Funzione per gestire le opzioni della linea di comando
int parse_args(int argc, char **argv, std::string &user, FacialAuthConfig &cfg) {
    int opt;
    while ((opt = getopt(argc, argv, "u:c:d:w:h:n:s:fvH")) != -1) {
        switch (opt) {
            case 'u':
                user = optarg;
                break;
            case 'c':
                cfg.model_path = optarg;
                break;
            case 'd':
                cfg.device = optarg;
                break;
            case 'w':
                cfg.width = std::stoi(optarg);
                break;
            case 'h':
                cfg.height = std::stoi(optarg);
                break;
            case 'n':
                cfg.frames = std::stoi(optarg);
                break;
            case 's':
                cfg.timeout = std::stoi(optarg); // Usa il timeout per la pausa
                break;
            case 'f':
                // Implementa la logica per forzare la sovrascrittura delle immagini esistenti
                break;
            case 'v':
                std::cout << "Verbose output enabled." << std::endl;
                break;
            case 'H':
            case '--help':
                print_usage(argv[0]);
                return -1;
            case 'z':
                // Puoi aggiungere altre opzioni come --debug o --nogui qui
                break;
            default:
                print_usage(argv[0]);
                return -1;
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    std::string user;
    FacialAuthConfig cfg;

    // Leggi e analizza i parametri dalla linea di comando
    if (parse_args(argc, argv, user, cfg) != 0) {
        return -1;
    }

    // Se non è stato fornito un nome utente, mostra un messaggio di errore
    if (user.empty()) {
        std::cerr << "Errore: Devi specificare un nome utente con -u <nome>" << std::endl;
        return -1;
    }

    // Crea un'istanza di FaceRecWrapper per l'addestramento (tipo modello: LBPH)
    FaceRecWrapper faceRec(cfg.model_path, "root", "LBPH");

    // Aggiungi la logica per la cattura della fotocamera e il riconoscimento del volto
    cv::VideoCapture cap(cfg.device);
    if (!cap.isOpened()) {
        std::cerr << "Impossibile aprire la fotocamera." << std::endl;
        return -1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, cfg.width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.height);

    // Carica il rilevamento del volto
    cv::CascadeClassifier haar;
    cv::dnn::Net dnn;
    bool use_dnn = false;
    std::string log;
    load_detectors(cfg, haar, dnn, use_dnn, log);

    // Ciclo di acquisizione frame dalla fotocamera
    cv::Mat frame;
    cv::Rect faceROI;
    int image_count = 0;

    while (image_count < cfg.frames) {
        cap >> frame;  // Acquisisci il frame
        if (frame.empty()) {
            std::cerr << "Errore nell'acquisizione del frame." << std::endl;
            break;
        }

        // Rilevamento del volto
        if (faceRec.DetectFace(cfg, frame, faceROI, haar, dnn)) {
            // Disegna un rettangolo attorno al volto rilevato
            cv::rectangle(frame, faceROI, cv::Scalar(255, 0, 0), 2);

            // Estrai la regione del volto e applica il riconoscimento
            cv::Mat face = frame(faceROI);
            int label = -1;
            double confidence = 0.0;
            faceRec.Predict(face, label, confidence);  // Riconosci il volto
            std::cout << "Utente: " << label << " con confidenza: " << confidence << std::endl;

            // Salva l'immagine del volto
            std::string filename = "/path/to/save/images/" + user + "_face_" + std::to_string(image_count) + ".jpg";
            cv::imwrite(filename, face);

            // Aumenta il contatore delle immagini
            image_count++;

            // Pausa tra una cattura e l'altra
            sleep_ms(cfg.timeout);
        }

        // Mostra il frame con il volto rilevato
        if (!cfg.nogui) {
            cv::imshow("Rilevamento Volto", frame);
        }

        // Uscita se premi 'q'
        char c = (char)cv::waitKey(1);
        if (c == 'q') break;
    }

    // Rilascia la fotocamera e chiudi le finestre
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
