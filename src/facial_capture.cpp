#include "../include/libfacialauth.h"
#include <iostream>
#include <filesystem>
#include <getopt.h>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <thread>
#include <chrono>

namespace fs = std::filesystem;

void print_help(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " -u <user> [options]\n\n"
    << "Options:\n"
    << "  -u, --user <name>       Nome utente per cui salvare le immagini\n"
    << "  -c, --config <file>     File di configurazione (default: " << FACIALAUTH_DEFAULT_CONFIG << ")\n"
    << "  -d, --device <path>     Device della webcam (es: /dev/video0)\n"
    << "  -w, --width <px>        Larghezza frame\n"
    << "  -h, --height <px>       Altezza frame\n"
    << "  -f, --force             Sovrascrive immagini esistenti e riparte da 1\n"
    << "  --flush, --clean        Elimina tutte le immagini per l'utente specificato\n"
    << "  -n, --num_images <num>  Numero di immagini da acquisire\n"
    << "  -s, --sleep <sec>       Pausa tra una cattura e l'altra (in secondi)\n"
    << "  -v, --verbose           Output dettagliato\n"
    << "  --debug                 Abilita output di debug\n"
    << "  --nogui                 Disabilita GUI, cattura solo da console\n"
    << "  --help, -H              Mostra questo messaggio\n";
}

int main(int argc, char** argv) {
    FacialAuthConfig cfg;
    std::string user;
    std::string config_path = FACIALAUTH_DEFAULT_CONFIG;
    bool force = false;
    bool flush = false;
    bool nogui = false;
    bool verbose = false;

    static struct option long_options[] = {
        {"user",       required_argument, 0, 'u'},
        {"config",     required_argument, 0, 'c'},
        {"device",     required_argument, 0, 'd'},
        {"width",      required_argument, 0, 'w'},
        {"height",     required_argument, 0, 'h'},
        {"force",      no_argument,       0, 'f'},
        {"flush",      no_argument,       0, 'C'}, // 'C' per Clean/Flush
        {"clean",      no_argument,       0, 'C'},
        {"num_images", required_argument, 0, 'n'},
        {"sleep",      required_argument, 0, 's'},
        {"verbose",    no_argument,       0, 'v'},
        {"debug",      no_argument,       0, 'D'},
        {"nogui",      no_argument,       0, 'G'},
        {"help",       no_argument,       0, 'H'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "u:c:d:w:h:fn:s:vvHG", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'u': user = optarg; break;
            case 'c': config_path = optarg; break;
            case 'd': cfg.device = optarg; break;
            case 'w': cfg.width = std::stoi(optarg); break;
            case 'h': cfg.height = std::stoi(optarg); break;
            case 'f': force = true; break;
            case 'C': flush = true; break;
            case 'n': cfg.frames = std::stoi(optarg); break;
            case 's': cfg.sleep_ms = static_cast<int>(std::stod(optarg) * 1000); break;
            case 'v': verbose = true; break;
            case 'D': cfg.debug = true; break;
            case 'G': nogui = true; break;
            case 'H': print_help(argv[0]); return 0;
            default: return 1;
        }
    }

    if (user.empty()) {
        std::cerr << "Errore: il parametro --user è obbligatorio.\n";
        print_help(argv[0]);
        return 1;
    }

    // Carica config se esiste, ma i parametri passati da CLI hanno priorità (già impostati sopra)
    std::string log;
    fa_load_config(cfg, log, config_path);

    std::string user_dir = cfg.basedir + "/" + user + "/captures";

    if (flush) {
        if (verbose) std::cout << "Eliminazione immagini esistenti in: " << user_dir << "\n";
        fs::remove_all(user_dir);
    }

    if (!fs::exists(user_dir)) {
        fs::create_directories(user_dir);
    }

    cv::VideoCapture cap;
    try {
        if (cfg.device.find_first_not_of("0123456789") == std::string::npos)
            cap.open(std::stoi(cfg.device));
        else
            cap.open(cfg.device);
    } catch (...) {
        cap.open(cfg.device);
    }

    if (!cap.isOpened()) {
        std::cerr << "Errore: Impossibile aprire la camera " << cfg.device << "\n";
        return 1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, cfg.width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.height);

    if (verbose) {
        std::cout << "User: " << user << "\n"
        << "Device: " << cfg.device << " (" << cfg.width << "x" << cfg.height << ")\n"
        << "Frames: " << cfg.frames << "\n";
    }

    int count = 0;
    int start_index = force ? 0 : 0; // Se non force, potremmo contare file esistenti
    if (!force) {
        for (auto const& dir_entry : fs::directory_iterator{user_dir}) count++;
    }

    while (count < cfg.frames) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        if (!nogui) {
            cv::imshow("Facial Capture - " + user, frame);
            if (cv::waitKey(1) == 'q') break;
        }

        std::string filename = user_dir + "/img_" + std::to_string(count) + "." + cfg.image_format;
        if (cv::imwrite(filename, frame)) {
            if (verbose) std::cout << "Salvato: " << filename << "\n";
            count++;
        }

        if (cfg.sleep_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(cfg.sleep_ms));
        }
    }

    std::cout << "Cattura completata. Totale immagini in " << user_dir << ": " << count << "\n";
    return 0;
}
