#include <iostream>
#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>
#include <unistd.h>

namespace fs = std::filesystem;

struct Config {
    std::string device = "/dev/video0";
    int width = 640;
    int height = 480;
};

Config load_config(const std::string &config_path, bool verbose) {
    Config cfg;
    std::ifstream conf(config_path);
    if (!conf.is_open()) {
        if (verbose)
            std::cerr << "[WARN] Config file not found, using defaults: " << config_path << std::endl;
        return cfg;
    }

    std::string key, value;
    while (conf >> key >> value) {
        if (key == "device") cfg.device = value;
        else if (key == "width") cfg.width = std::stoi(value);
        else if (key == "height") cfg.height = std::stoi(value);
    }

    if (verbose) {
        std::cout << "[INFO] Loaded config from " << config_path << std::endl;
        std::cout << "[INFO] Device: " << cfg.device << ", width=" << cfg.width << ", height=" << cfg.height << std::endl;
    }
    return cfg;
}

void flush_images(const std::string &user_dir, const std::string &user, bool verbose) {
    fs::path dir_path = user_dir + "/" + user;
    if (!fs::exists(dir_path)) {
        if (verbose)
            std::cerr << "[WARN] Directory not found for user: " << user << std::endl;
        return;
    }

    try {
        for (const auto &entry : fs::directory_iterator(dir_path)) {
            if (entry.is_regular_file()) {
                fs::remove(entry);
                if (verbose)
                    std::cout << "[INFO] Removed: " << entry.path() << std::endl;
            }
        }
    } catch (const fs::filesystem_error &e) {
        std::cerr << "[ERROR] Error deleting images: " << e.what() << std::endl;
    }

    std::cout << "[INFO] All images for user " << user << " have been deleted." << std::endl;
}

std::string get_next_filename(const std::string &user_dir, const std::string &user, bool force, bool verbose) {
    if (force) {
        if (verbose)
            std::cout << "[DEBUG] Force mode active: starting numbering from 1" << std::endl;
        return user_dir + "/" + user + "_1.jpg";
    }

    int index = 1;
    std::string filename;
    do {
        filename = user_dir + "/" + user + "_" + std::to_string(index) + ".jpg";
        index++;
    } while (fs::exists(filename));
    return filename;
}

int main(int argc, char **argv) {
    std::string user;
    std::string config_path = "/etc/pam_facial_auth/pam_facial.conf";
    bool verbose = false;
    bool force = false;
    bool flush = false;
    bool nogui = false;
    std::string device_override;
    int width_override = -1, height_override = -1;

    // Parse args
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-u" || arg == "--user") && i + 1 < argc) {
            user = argv[++i];
        } else if ((arg == "-c" || arg == "--config") && i + 1 < argc) {
            config_path = argv[++i];
        } else if ((arg == "-d" || arg == "--device") && i + 1 < argc) {
            device_override = argv[++i];
        } else if ((arg == "-w" || arg == "--width") && i + 1 < argc) {
            width_override = std::stoi(argv[++i]);
        } else if ((arg == "-h" || arg == "--height") && i + 1 < argc) {
            height_override = std::stoi(argv[++i]);
        } else if (arg == "-f" || arg == "--force") {
            force = true;
        } else if (arg == "--flush" || arg == "--clean") {
            flush = true;
        } else if (arg == "-v" || arg == "--verbose") {
            verbose = true;
        } else if (arg == "--nogui") {
            nogui = true;
        } else if (arg == "--help" || arg == "-H") {
            std::cout << "Usage: facial_capture -u <user> [options]\n\n"
            << "Options:\n"
            << "  -u, --user <name>       Nome utente per cui salvare le immagini\n"
            << "  -c, --config <file>     File di configurazione (default: /etc/pam_facial_auth/pam_facial.conf)\n"
            << "  -d, --device <path>     Device della webcam (es: /dev/video0)\n"
            << "  -w, --width <px>        Larghezza frame\n"
            << "  -h, --height <px>       Altezza frame\n"
            << "  -f, --force             Sovrascrive immagini esistenti e riparte da 1\n"
            << "  --flush, --clean        Elimina tutte le immagini per l'utente specificato\n"
            << "  -v, --verbose           Output dettagliato\n"
            << "  --nogui                 Disabilita la GUI, usa solo la console\n"
            << "  --help, -H              Mostra questo messaggio\n";
            return 0;
        }
    }

    if (user.empty()) {
        std::cerr << "[ERROR] Devi specificare un utente con -u <nome>\n";
        return 1;
    }

    if (flush) {
        flush_images("/etc/pam_facial_auth", user, verbose);
        return 0;
    }

    // Carica la configurazione
    Config cfg = load_config(config_path, verbose);
    if (!device_override.empty()) cfg.device = device_override;
    if (width_override > 0) cfg.width = width_override;
    if (height_override > 0) cfg.height = height_override;

    // Directory per le immagini dell'utente
    std::string user_dir = "/etc/pam_facial_auth/" + user;
    if (!fs::exists(user_dir)) {
        if (verbose) std::cout << "[INFO] Creazione directory utente: " << user_dir << std::endl;
        fs::create_directories(user_dir);
    }

    // Apri la webcam
    cv::VideoCapture cap(cfg.device);
    if (!cap.isOpened()) {
        std::cerr << "[ERROR] Impossibile aprire il dispositivo: " << cfg.device << std::endl;
        return 1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, cfg.width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.height);

    if (verbose)
        std::cout << "[INFO] Webcam " << cfg.device << " aperta (" << cfg.width << "x" << cfg.height << ")\n"
        << "[INFO] Premi 's' per salvare, 'q' per uscire.\n";

    cv::Mat frame;
    int saved_count = 0;

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "[ERROR] Frame non valido.\n";
            break;
        }

        if (!nogui) {
            cv::imshow("Facial Capture - Premere 's' per salvare, 'q' per uscire", frame);
        }

        char key = (char)cv::waitKey(1);

        if (key == 's') {
            std::string filename = get_next_filename(user_dir, user, force, verbose);
            cv::imwrite(filename, frame);
            saved_count++;
            if (verbose) std::cout << "[INFO] Salvata immagine: " << filename << std::endl;
            if (force) {
                if (verbose) std::cout << "[DEBUG] Force attivo, sovrascrittura terminata dopo 1 immagine.\n";
                break;
            }
        } else if (key == 'q') {
            if (verbose) std::cout << "[INFO] Uscita richiesta dall'utente.\n";
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    if (verbose)
        std::cout << "[INFO] Capture terminato. Immagini salvate: " << saved_count << std::endl;

    return 0;
}
