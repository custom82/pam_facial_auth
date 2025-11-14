#include "../include/libfacialauth.h"
#include <iostream>
#include <filesystem>
#include <unistd.h>

using namespace std;
namespace fs = std::filesystem;

static void show_help() {
    cout << "Usage: facial_capture [options]\n\n"
    << "Options:\n"
    << "  -u, --user <name>        Nome utente (obbligatorio)\n"
    << "  -d, --device <dev>       Dispositivo video (default: /dev/video0)\n"
    << "  --width <px>             Larghezza frame\n"
    << "  --height <px>            Altezza frame\n"
    << "  --sleep <ms>             Millisecondi tra uno scatto e l'altro\n"
    << "  --debug                  Modalità debug\n"
    << "  --nogui                  Disabilita GUI\n"
    << "  --list-devices           Mostra i device video disponibili\n"
    << "  -f, --force              Sovrascrive immagini esistenti\n"
    << "  -h, --help               Mostra questo messaggio\n"
    << endl;
}

static void list_devices() {
    cout << "Video devices disponibili:\n";
    for (int i = 0; i < 10; i++) {
        string dev = "/dev/video" + to_string(i);
        if (fs::exists(dev))
            cout << "  " << dev << endl;
    }
}

int main(int argc, char *argv[]) {
    string user;
    string device = "/dev/video0";
    bool force = false;

    FacialAuthConfig cfg;

    // ------------------------
    // PARSING ARGOMENTI
    // ------------------------
    for (int i = 1; i < argc; i++) {
        string a = argv[i];

        if (a == "-u" || a == "--user") {
            user = argv[++i];
        }
        else if (a == "-d" || a == "--device") {
            device = argv[++i];
        }
        else if (a == "--width") {
            cfg.frame_width = stoi(argv[++i]);
        }
        else if (a == "--height") {
            cfg.frame_height = stoi(argv[++i]);
        }
        else if (a == "--sleep") {
            cfg.sleep_ms = stoi(argv[++i]);
        }
        else if (a == "--debug") {
            cfg.debug = true;
        }
        else if (a == "--nogui") {
            cfg.nogui = true;
        }
        else if (a == "--list-devices") {
            list_devices();
            return 0;
        }
        else if (a == "-f" || a == "--force") {
            force = true;
        }
        else if (a == "-h" || a == "--help") {
            show_help();
            return 0;
        }
    }

    if (user.empty()) {
        cerr << "ERRORE: specificare un utente con -u\n";
        return 1;
    }

    // Preparazione directory
    string user_dir = "/etc/pam_facial_auth/" + user + "/images";
    if (fs::exists(user_dir) && !force) {
        cerr << "ERRORE: directory " << user_dir
        << " esiste già. Usa --force per sovrascrivere.\n";
        return 1;
    }

    fs::create_directories(user_dir);

    // ------------------------
    // CATTURA TRAMITE LIB
    // ------------------------
    FaceRecWrapper faceRec;

    cout << "Inizio cattura immagini...\n";
    if (!faceRec.CaptureImages(user, cfg)) {
        cerr << "Errore durante la cattura immagini.\n";
        return 1;
    }

    cout << "Cattura completata.\n";
    return 0;
}
