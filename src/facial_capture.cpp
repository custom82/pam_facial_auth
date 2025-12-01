#include "libfacialauth.h"

#include <opencv2/core.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <fstream>
#include <filesystem>


#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

static void print_help()
{
    std::cout <<
    "Usage: facial_capture -u <user> [options]\n\n"
    "Options:\n"
    "  -u, --user <name>          Nome utente per cui salvare le immagini\n"
    "  -c, --config <file>        File di configurazione\n"
    "                             (default: /etc/pam_facial_auth/pam_facial.conf)\n"
    "  -d, --device <path>        Device della webcam (es: /dev/video0)\n"
    "  -w, --width <px>           Larghezza frame\n"
    "  -h, --height <px>          Altezza frame\n"
    "  -f, --force                Sovrascrive immagini esistenti\n"
    "      --flush, --clean       Elimina tutte le immagini dell'utente e termina\n"
    "  -n, --num-images <num>     Numero di immagini da acquisire\n"
    "  -s, --sleep <sec>          Pausa tra una cattura e l'altra (secondi)\n"
    "      --detector <name>      Selettore del detector (haar|yunet|yunet_int8|auto)\n"
    "      --list-detectors       Elenca i detector disponibili dal file di config\n"
    "      --list-devices         Elenca le webcam disponibili (tramite V4L2)\n"
    "      --list-resolutions     Elenca le risoluzioni supportate dalla webcam\n"
    "      --format <fmt>         Formato immagine: jpg|png|bmp\n"
    "  -v, --verbose              Output dettagliato\n"
    "      --debug                Abilita output di debug\n"
    "      --nogui                Disabilita GUI, cattura solo da console\n"
    "  -H, --help                 Mostra questo messaggio\n";
}

// ----------------------------------------------------------
// Helpers V4L2
// ----------------------------------------------------------

static bool has_video_capture_capability(int fd)
{
    struct v4l2_capability cap {};
    if (ioctl(fd, VIDIOC_QUERYCAP, &cap) != 0)
        return false;

    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE) &&
        !(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE_MPLANE))
        return false;

    return true;
}

static std::string read_sysfs_attr(const std::string &path)
{
    std::ifstream f(path);
    if (!f.is_open())
        return {};
    std::string v;
    f >> v;
    return v;
}

static void list_devices_v4l()
{
    for (int i = 0; i < 32; ++i) {
        std::string dev = "/dev/video" + std::to_string(i);
        int fd = ::open(dev.c_str(), O_RDONLY | O_NONBLOCK);
        if (fd < 0)
            continue;

        struct v4l2_capability cap {};
        if (!has_video_capture_capability(fd)) {
            ::close(fd);
            continue;
        }

        if (ioctl(fd, VIDIOC_QUERYCAP, &cap) != 0) {
            ::close(fd);
            continue;
        }

        std::string card(reinterpret_cast<const char*>(cap.card));
        std::string bus(reinterpret_cast<const char*>(cap.bus_info));

        // Proviamo a recuperare vendor/product da sysfs (solo per dispositivi USB / PCI)
        std::string sysbase = "/sys/class/video4linux/video" + std::to_string(i) + "/device/";
        std::string vendor  = read_sysfs_attr(sysbase + "idVendor");
        std::string product = read_sysfs_attr(sysbase + "idProduct");

        std::cout << dev << "  ";

        if (!vendor.empty() && !product.empty())
            std::cout << "(" << vendor << ":" << product << ") ";

        std::cout << card;

        if (!bus.empty())
            std::cout << " [" << bus << "]";

        std::cout << "\n";

        ::close(fd);
    }
}

static void list_resolutions_for_device(const std::string &device)
{
    int fd = ::open(device.c_str(), O_RDONLY | O_NONBLOCK);
    if (fd < 0) {
        std::cerr << "[ERRORE] Impossibile aprire " << device << " per leggere le risoluzioni.\n";
        return;
    }

    if (!has_video_capture_capability(fd)) {
        std::cerr << "[ERRORE] Il device " << device << " non supporta la cattura video.\n";
        ::close(fd);
        return;
    }

    std::cout << "Risoluzioni disponibili per " << device << ":\n";

    struct v4l2_fmtdesc fmtdesc {};
    fmtdesc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    for (fmtdesc.index = 0; ioctl(fd, VIDIOC_ENUM_FMT, &fmtdesc) == 0; ++fmtdesc.index) {
        uint32_t pixelformat = fmtdesc.pixelformat;
        char fourcc[5] = {
            char(pixelformat & 0xFF),
            char((pixelformat >> 8) & 0xFF),
            char((pixelformat >> 16) & 0xFF),
            char((pixelformat >> 24) & 0xFF),
            0
        };

        std::cout << "  Format " << fmtdesc.index << " (" << fourcc << "):\n";

        struct v4l2_frmsizeenum frmsize {};
        frmsize.pixel_format = pixelformat;

        for (frmsize.index = 0; ioctl(fd, VIDIOC_ENUM_FRAMESIZES, &frmsize) == 0; ++frmsize.index) {
            if (frmsize.type == V4L2_FRMSIZE_TYPE_DISCRETE) {
                std::cout << "    "
                << frmsize.discrete.width  << "x"
                << frmsize.discrete.height << "\n";
            } else if (frmsize.type == V4L2_FRMSIZE_TYPE_STEPWISE ||
                frmsize.type == V4L2_FRMSIZE_TYPE_CONTINUOUS) {
                std::cout << "    "
                << frmsize.stepwise.min_width << "x" << frmsize.stepwise.min_height
                << " .. "
                << frmsize.stepwise.max_width << "x" << frmsize.stepwise.max_height
                << "\n";
            break; // Evitiamo spam infinito
                }
        }
    }

    ::close(fd);
}

// ----------------------------------------------------------
// Helpers per config / detectors
// ----------------------------------------------------------

static void list_detectors_from_config(const FacialAuthConfig &cfg)
{
    if (cfg.detector_models.empty()) {
        std::cout << "Nessun detector configurato (chiavi detect_* mancanti nel file di config).\n";
        return;
    }

    std::cout << "Detector disponibili (da file di configurazione):\n";
    for (const auto &kv : cfg.detector_models) {
        std::cout << "  " << kv.first;
        if (!kv.second.empty())
            std::cout << " -> " << kv.second;
        std::cout << "\n";
    }
}

// Normalizza il nome passato a --detector:
//  - se inizia con "detect_" lo taglia
//  - altrimenti usa il nome così com'è
static std::string normalize_detector_name(const std::string &name)
{
    if (name.rfind("detect_", 0) == 0)
        return name.substr(std::string("detect_").size());
    return name;
}

static bool apply_detector_override(FacialAuthConfig &cfg,
                                    const std::string &det_name,
                                    std::string &log)
{
    if (det_name.empty())
        return true;

    std::string norm = normalize_detector_name(det_name);

    // accettiamo alias classici
    if (norm == "haar" || norm == "yunet" || norm == "yunet_fp32" || norm == "yunet_int8") {
        cfg.detector_profile = norm;
        return true;
    }

    // Se nel map detector_models esiste una voce con quel nome, usiamo quel profilo
    auto it = cfg.detector_models.find(norm);
    if (it != cfg.detector_models.end()) {
        cfg.detector_profile = norm;
        return true;
    }

    log += "Detector sconosciuto: " + det_name + "\n";
    return false;
}

// ----------------------------------------------------------
// Implementazione CLI principale
// ----------------------------------------------------------

int facial_capture_main(int argc, char **argv)
{
    std::string user;
    std::string config_path = "/etc/pam_facial_auth/pam_facial.conf";
    std::string device_override;
    int width_override  = -1;
    int height_override = -1;
    int frames_override = -1;   // numero immagini
    int sleep_sec       = -1;
    std::string detector_override;
    std::string format_override;
    bool force      = false;
    bool flush_only = false;
    bool verbose    = false;
    bool debug      = false;
    bool nogui      = false;
    bool list_devices    = false;
    bool list_res        = false;
    bool list_detectors  = false;

    // ----------------- Parsing argomenti -----------------
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        auto take_value = [&](const std::string &opt) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "Manca il valore per l'opzione " << opt << "\n";
                exit(1);
            }
            return argv[++i];
        };

        if (arg == "-u" || arg == "--user") {
            user = take_value(arg);
        } else if (arg == "-c" || arg == "--config") {
            config_path = take_value(arg);
        } else if (arg == "-d" || arg == "--device") {
            device_override = take_value(arg);
        } else if (arg == "-w" || arg == "--width") {
            width_override = std::stoi(take_value(arg));
        } else if (arg == "-h" || arg == "--height") {
            height_override = std::stoi(take_value(arg));
        } else if (arg == "-f" || arg == "--force") {
            force = true;
        } else if (arg == "--flush" || arg == "--clean") {
            flush_only = true;
        } else if (arg == "-n" || arg == "--num-images") {
            frames_override = std::stoi(take_value(arg));
        } else if (arg == "-s" || arg == "--sleep") {
            sleep_sec = std::stoi(take_value(arg));
        } else if (arg == "--detector") {
            detector_override = take_value(arg);
        } else if (arg == "--list-detectors") {
            list_detectors = true;
        } else if (arg == "--list-devices") {
            list_devices = true;
        } else if (arg == "--list-resolutions") {
            list_res = true;
        } else if (arg == "--format") {
            format_override = take_value(arg);
        } else if (arg == "-v" || arg == "--verbose") {
            verbose = true;
        } else if (arg == "--debug") {
            debug = true;
        } else if (arg == "--nogui") {
            nogui = true;
        } else if (arg == "--help" || arg == "-H") {
            print_help();
            return 0;
        } else {
            std::cerr << "Opzione sconosciuta: " << arg << "\n";
            print_help();
            return 1;
        }
    }

    // Opzioni che non richiedono config/utente
    if (list_devices) {
        list_devices_v4l();
        return 0;
    }

    // Carichiamo la configurazione
    FacialAuthConfig cfg;
    std::string log;
    if (!fa_load_config(cfg, log, config_path)) {
        std::cerr << log;
        // continuiamo con i defaults di cfg (già inizializzato nel costruttore)
    }

    // Applichiamo override da CLI
    if (!device_override.empty())
        cfg.device = device_override;
    if (width_override > 0)
        cfg.width = width_override;
    if (height_override > 0)
        cfg.height = height_override;
    if (frames_override > 0)
        cfg.frames = frames_override;
    if (sleep_sec >= 0)
        cfg.sleep_ms = sleep_sec * 1000;
    if (!format_override.empty())
        cfg.image_format = format_override;

    if (verbose)
        cfg.debug = true;
    if (debug)
        cfg.debug = true;
    if (nogui)
        cfg.nogui = true;
    if (force)
        cfg.force_overwrite = true;

    // Validazione formato
    {
        std::string low = cfg.image_format;
        std::transform(low.begin(), low.end(), low.begin(),
                       [](unsigned char c){ return std::tolower(c); });
        if (low != "jpg" && low != "jpeg" && low != "png" && low != "bmp") {
            std::cerr << "[ERRORE] Formato immagine non supportato: " << cfg.image_format
            << " (usa jpg|png|bmp).\n";
            return 1;
        }
        if (low == "jpeg")
            cfg.image_format = "jpg";
    }

    // Lista detectors (richiede config)
    if (list_detectors) {
        list_detectors_from_config(cfg);
        return 0;
    }

    // Lista risoluzioni (usa device da config/override)
    if (list_res) {
        list_resolutions_for_device(cfg.device);
        return 0;
    }

    // Flush: cancella immagini e termina
    if (flush_only) {
        std::string imgdir = fa_user_image_dir(cfg, user.empty() ? "default" : user);
        // Usa helper di libfacialauth per creare la dir se serve, ma qui la svuotiamo a mano
        if (user.empty()) {
            std::cerr << "[ERRORE] --flush richiede anche --user.\n";
            return 1;
        }

        namespace fs = std::filesystem;
        try {
            fs::path p(imgdir);
            if (fs::exists(p) && fs::is_directory(p)) {
                for (auto &entry : fs::directory_iterator(p)) {
                    fs::remove(entry.path());
                }
                std::cout << "[INFO] Immagini cancellate in " << imgdir << "\n";
            } else {
                std::cout << "[INFO] Nessuna immagine da cancellare (directory inesistente).\n";
            }
        } catch (const std::exception &e) {
            std::cerr << "[ERRORE] Impossibile cancellare le immagini: " << e.what() << "\n";
            return 1;
        }
        return 0;
    }

    // Da qui in poi, richiediamo lo user
    if (user.empty()) {
        std::cerr << "[ERRORE] Devi specificare --user <name>.\n";
        return 1;
    }

    // Applicazione override detector (--detector)
    if (!detector_override.empty()) {
        if (!apply_detector_override(cfg, detector_override, log)) {
            std::cerr << log;
            return 1;
        }
    }

    // Verifica privilegi se necessario
    if (!fa_check_root("facial_capture")) {
        std::cerr << "[ERRORE] Questo strumento deve essere eseguito come root.\n";
        return 1;
    }

    // Chiamata alla libreria: cattura immagini con rilevamento volto
    std::string capture_log;
    if (!fa_capture_images(user, cfg, cfg.image_format, capture_log)) {
        std::cerr << capture_log;
        return 1;
    }

    if (!capture_log.empty())
        std::cout << capture_log;

    return 0;
}

// ----------------------------------------------------------
// Wrapper main() con gestione eccezioni OpenCV/std
// ----------------------------------------------------------

int main(int argc, char **argv)
{
    try {
        return facial_capture_main(argc, argv);
    }
    catch (const cv::Exception &e) {
        std::cerr << "[OpenCV ERROR] " << e.what() << std::endl;
        return 1;
    }
    catch (const std::exception &e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 1;
    }
}
