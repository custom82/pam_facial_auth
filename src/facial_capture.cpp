#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <chrono>
#include <thread>
#include <sstream>
#include <vector>

namespace fs = std::filesystem;

struct Config {
    int width = 640;
    int height = 480;
};

// Funzione per leggere width e height dal file di configurazione
void loadConfig(const std::string &configPath, Config &cfg, bool verbose) {
    std::ifstream file(configPath);
    if (!file.is_open()) {
        if (verbose) std::cerr << "[WARN] Impossibile aprire il file di configurazione: " << configPath << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.find("width=") == 0)
            cfg.width = std::stoi(line.substr(6));
        else if (line.find("height=") == 0)
            cfg.height = std::stoi(line.substr(7));
    }
    file.close();
}

// Funzione per generare il nome dell'immagine e verificare la sequenza
std::string generateFileName(const std::string& userDir, const std::string& user, bool force) {
    int counter = 0;
    std::string filename;
    do {
        filename = userDir + "/" + user + "_" + std::to_string(counter++) + ".jpg";
    } while (fs::exists(filename) && !force);
    return filename;
}

// Mostra l’help
void showHelp() {
    std::cout << "Usage: facial_capture [options]\n\n"
    << "Options:\n"
    << "  -u, --user <name>       Nome utente (obbligatorio)\n"
    << "  -d, --device <path>     Dispositivo webcam (default: /dev/video0)\n"
    << "  -o, --output <dir>      Directory di output per le immagini\n"
    << "  -n, --num <N>           Numero di immagini da catturare (default: 30)\n"
    << "  --width <W>             Larghezza fotogramma (override da config)\n"
    << "  --height <H>            Altezza fotogramma (override da config)\n"
    << "  -f, --force             Sovrascrivi le immagini esistenti\n"
    << "  --nogui                 Cattura senza mostrare anteprima\n"
    << "  -v, --verbose           Modalità verbosa\n"
    << "  -h, --help              Mostra questo messaggio\n";
}

int main(int argc, char **argv) {
    std::string user;
    std::string device = "/dev/video0";
    std::string outputDir;
    std::string configPath = "/etc/pam_facial_auth/pam_facial.conf";
    int numImages = 30;
    bool verbose = false;
    bool nogui = false;
    bool force = false;
    Config cfg;

    // Carica valori da file di configurazione
    loadConfig(configPath, cfg, verbose);

    // Parsing manuale argomenti
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-u" || arg == "--user") && i + 1 < argc) user = argv[++i];
        else if ((arg == "-d" || arg == "--device") && i + 1 < argc) device = argv[++i];
        else if ((arg == "-o" || arg == "--output") && i + 1 < argc) outputDir = argv[++i];
        else if ((arg == "-n" || arg == "--num") && i + 1 < argc) numImages = std::stoi(argv[++i]);
        else if ((arg == "--width" || arg == "--height") && i + 1 < argc) {
            if (arg == "--width") cfg.width = std::stoi(argv[++i]);
            if (arg == "--height") cfg.height = std::stoi(argv[++i]);
        }
        else if (arg == "--nogui") nogui = true;
        else if (arg == "-f" || arg == "--force") force = true;
        else if (arg == "-v" || arg == "--verbose") verbose = true;
        else if (arg == "-h" || arg == "--help") { showHelp(); return 0; }
        else {
            std::cerr << "Parametro sconosciuto: " << arg << std::endl;
            showHelp();
            return 1;
        }
    }

    if (user.empty() || outputDir.empty()) {
        std::cerr << "[ERRORE] Parametri mancanti. Usa --help per maggiori informazioni.\n";
        return 1;
    }

    if (verbose) {
        std::cout << "[INFO] Utente: " << user << "\n"
        << "[INFO] Device: " << device << "\n"
        << "[INFO] Output dir: " << outputDir << "\n"
        << "[INFO] Risoluzione: " << cfg.width << "x" << cfg.height << "\n"
        << "[INFO] Numero immagini: " << numImages << "\n";
    }

    // Crea la directory di output se non esiste
    fs::create_directories(outputDir + "/" + user);

    cv::VideoCapture cap(device);
    if (!cap.isOpened()) {
        std::cerr << "[ERRORE] Impossibile aprire la webcam: " << device << std::endl;
        return 1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, cfg.width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.height);

    cv::Mat frame;
    for (int i = 0; i < numImages; ++i) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "[WARN] Frame vuoto, salto...\n";
            continue;
        }

        std::string userDir = outputDir + "/" + user;
        std::string filename = generateFileName(userDir, user, force);

        cv::imwrite(filename, frame);
        if (verbose)
            std::cout << "[DEBUG] Immagine salvata: " << filename << std::endl;

        if (!nogui) {
            cv::imshow("Facial Capture", frame);
            if (cv::waitKey(500) == 27) break; // ESC per uscire
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }

    cap.release();
    if (!nogui) cv::destroyAllWindows();
    std::cout << "[OK] Acquisizione completata con successo." << std::endl;

    return 0;
}
