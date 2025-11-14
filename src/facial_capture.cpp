#include "../include/libfacialauth.h"

#include <getopt.h>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

static void print_usage(const char *prog) {
    std::cerr
    << "Usage: " << prog << " -u <user> [options]\n\n"
    << "Options:\n"
    << "  -u, --user <name>       Nome utente (obbligatorio)\n"
    << "  -c, --config <file>     File di configurazione (default: /etc/security/pam_facial.conf)\n"
    << "  -d, --device <path>     Device della webcam (es: /dev/video0)\n"
    << "      --width <px>        Larghezza frame\n"
    << "      --height <px>       Altezza frame\n"
    << "  -n, --num-images <num>  Numero di immagini da acquisire\n"
    << "  -s, --sleep <sec>       Pausa tra una cattura e l'altra (secondi)\n"
    << "  -f, --force             Sovrascrive immagini esistenti e riparte da 1\n"
    << "      --flush             Elimina tutte le immagini per l'utente specificato\n"
    << "      --list-devices      Elenca /dev/video[0-9]\n"
    << "      --nogui             Disabilita GUI\n"
    << "      --debug             Abilita debug\n"
    << "  -v, --verbose           Output dettagliato\n"
    << "  -h, --help              Mostra questo messaggio\n";
}

static void list_devices() {
    std::cout << "Possible V4L devices:\n";
    for (int i = 0; i < 10; ++i) {
        std::string path = "/dev/video" + std::to_string(i);
        if (fs::exists(path)) std::cout << "  " << path << "\n";
    }
}

int main(int argc, char **argv) {
    std::string user;
    std::string config_path = "/etc/security/pam_facial.conf";
    bool force = false;
    bool flush = false;
    bool verbose = false;
    bool listdev = false;

    FacialAuthConfig cfg; // con default

    static struct option long_opts[] = {
        {"user",          required_argument, 0, 'u'},
        {"config",        required_argument, 0, 'c'},
        {"device",        required_argument, 0, 'd'},
        {"width",         required_argument, 0, 1},
        {"height",        required_argument, 0, 2},
        {"num-images",    required_argument, 0, 'n'},
        {"sleep",         required_argument, 0, 3},
        {"force",         no_argument,       0, 'f'},
        {"flush",         no_argument,       0, 4},
        {"clean",         no_argument,       0, 4},
        {"list-devices",  no_argument,       0, 5},
        {"nogui",         no_argument,       0, 6},
        {"debug",         no_argument,       0, 7},
        {"verbose",       no_argument,       0, 'v'},
        {"help",          no_argument,       0, 'h'},
        {0,0,0,0}
    };

    int opt, idx;
    while ((opt = getopt_long(argc, argv, "u:c:d:n:s:fvh", long_opts, &idx)) != -1) {
        switch (opt) {
            case 'u': user = optarg; break;
            case 'c': config_path = optarg; break;
            case 'd': cfg.device = optarg; break;
            case 'n': cfg.frames = std::max(1, std::stoi(optarg)); break;
            case 's': cfg.sleep_ms = std::max(0, std::stoi(optarg) * 1000); break;
            case 'f': force = true; break;
            case 'v': verbose = true; break;
            case 'h': print_usage(argv[0]); return 0;
            case 1:   cfg.width  = std::max(64, std::stoi(optarg)); break;
            case 2:   cfg.height = std::max(64, std::stoi(optarg)); break;
            case 3:   cfg.sleep_ms = std::max(0, std::stoi(optarg) * 1000); break;
            case 4:   flush = true; break;
            case 5:   listdev = true; break;
            case 6:   cfg.nogui = true; break;
            case 7:   cfg.debug = true; break;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }

    if (listdev) {
        list_devices();
        return 0;
    }

    if (user.empty()) {
        std::cerr << "Errore: devi specificare un utente con -u <user>\n";
        print_usage(argv[0]);
        return 1;
    }

    std::string log;
    read_kv_config(config_path, cfg, &log);

    if (flush) {
        std::string dir = fa_user_image_dir(cfg, user);
        if (fs::exists(dir)) {
            fs::remove_all(dir);
            if (verbose) std::cout << "Removed directory: " << dir << "\n";
        } else {
            if (verbose) std::cout << "Directory not found: " << dir << "\n";
        }
        return 0;
    }

    bool ok = fa_capture_images(user, cfg, force, log);

    if (verbose || cfg.debug) std::cerr << log;
    return ok ? 0 : 1;
}
