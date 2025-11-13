#include <iostream>
#include <string>
#include <unistd.h>
#include <getopt.h>
#include <dirent.h>
#include "../include/libfacialauth.h"

// -----------------------------------------------------
// Stampa help
// -----------------------------------------------------
void print_usage(const char *prog) {
    std::cout << "Usage: " << prog << " -u <user> [options]\n\n"
    << "Options:\n"
    << "  -u, --user <name>         Nome utente\n"
    << "  -c, --config <file>       (non usato, riservato)\n"
    << "  -d, --device <path>       Device webcam (es: /dev/video0)\n"
    << "  -w, --width <px>          Larghezza frame\n"
    << "  -h, --height <px>         Altezza frame\n"
    << "  -n, --num-images <num>    Numero di immagini da catturare\n"
    << "  -s, --sleep <sec>         Pausa tra una cattura e l'altra\n"
    << "  -f, --force               Cancella immagini esistenti e ricomincia\n"
    << "      --clean               Cancella immagini e termina\n"
    << "      --nogui               Disabilita GUI\n"
    << "      --debug               Abilita log verbose\n"
    << "      --list-devices        Elenca i device video disponibili\n"
    << "      --help                Mostra questo messaggio\n";
}

// -----------------------------------------------------
// Elimina immagini esistenti in una directory
// -----------------------------------------------------
bool purge_user_images(const std::string &dir)
{
    DIR *dp = opendir(dir.c_str());
    if (!dp)
        return false;

    struct dirent *ent;
    while ((ent = readdir(dp)) != nullptr) {
        if (ent->d_name[0] == '.') continue;
        std::string path = join_path(dir, ent->d_name);
        unlink(path.c_str());
    }

    closedir(dp);
    return true;
}

// -----------------------------------------------------
// Elenca /dev/video*
// -----------------------------------------------------
void list_devices()
{
    std::cout << "Device video disponibili:\n";

    DIR *dp = opendir("/dev");
    if (!dp) {
        std::cerr << "Impossibile aprire /dev\n";
        return;
    }

    struct dirent *ent;
    bool any = false;
    while ((ent = readdir(dp)) != nullptr) {
        std::string name = ent->d_name;
        if (name.rfind("video", 0) == 0) {
            any = true;
            std::cout << "  /dev/" << name << "\n";
        }
    }
    closedir(dp);

    if (!any)
        std::cout << "  Nessun device video trovato.\n";
}

// -----------------------------------------------------
// Parsing parametri
// -----------------------------------------------------
int parse_args(int argc, char **argv,
               std::string &user, FacialAuthConfig &cfg,
               bool &force, bool &clean_only, bool &list_dev)
{
    static struct option long_opts[] = {
        {"user",          required_argument, nullptr, 'u'},
        {"config",        required_argument, nullptr, 'c'},
        {"device",        required_argument, nullptr, 'd'},
        {"width",         required_argument, nullptr, 'w'},
        {"height",        required_argument, nullptr, 'h'},
        {"num-images",    required_argument, nullptr, 'n'},
        {"sleep",         required_argument, nullptr, 's'},
        {"force",         no_argument,       nullptr, 'f'},
        {"clean",         no_argument,       nullptr,  4 },
        {"flush",         no_argument,       nullptr,  4 },
        {"nogui",         no_argument,       nullptr,  1 },
        {"debug",         no_argument,       nullptr,  2 },
        {"list-devices",  no_argument,       nullptr,  5 },
        {"help",          no_argument,       nullptr,  3 },
        {nullptr, 0, nullptr, 0}
    };

    int long_idx = 0;
    int opt;

    while ((opt = getopt_long(argc, argv, "u:c:d:w:h:n:s:f", long_opts, &long_idx)) != -1)
    {
        switch (opt) {
            case 'u':
                user = optarg;
                break;

            case 'c':
                // placeholder per futuro uso (file config extra)
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
                cfg.timeout = std::stoi(optarg);
                break;

            case 'f':
                force = true;
                break;

            case 4: // clean / flush
                clean_only = true;
                break;

            case 1: // nogui
                cfg.nogui = true;
                break;

            case 2: // debug
                cfg.debug = true;
                break;

            case 5: // list-devices
                list_dev = true;
                break;

            case 3: // help
                print_usage(argv[0]);
                return -1;

            default:
                print_usage(argv[0]);
                return -1;
        }
    }

    return 0;
}

// -----------------------------------------------------
// MAIN
// -----------------------------------------------------
int main(int argc, char **argv)
{
    std::string user;
    FacialAuthConfig cfg;
    bool force = false;
    bool clean_only = false;
    bool list_dev = false;

    // carica config di default, se presente
    read_kv_config("/etc/security/pam_facial.conf", cfg, nullptr);

    if (parse_args(argc, argv, user, cfg, force, clean_only, list_dev) != 0)
        return 1;

    if (list_dev) {
        list_devices();
        return 0;
    }

    if (user.empty()) {
        std::cerr << "Errore: -u/--user <name> è obbligatorio.\n";
        return 1;
    }

    // directory utente: <model_path>/<user>/images
    std::string user_dir = join_path(cfg.model_path, user);
    std::string img_dir  = join_path(user_dir, "images");
    ensure_dirs(img_dir);

    if (clean_only) {
        std::cout << "[INFO] Pulizia immagini in " << img_dir << "\n";
        purge_user_images(img_dir);
        return 0;
    }

    if (force) {
        std::cout << "[INFO] Forzatura attiva: cancello immagini esistenti in "
        << img_dir << "\n";
        purge_user_images(img_dir);
    }

    FaceRecWrapper faceRec(cfg.model_path, user, "LBPH");

    if (!faceRec.CaptureImages(user, cfg)) {
        std::cerr << "Errore nella cattura delle immagini.\n";
        return 1;
    }

    return 0;
}
