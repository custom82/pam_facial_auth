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
#include <fcntl.h>
#include <unistd.h>

static void print_help()
{
    std::cout <<
    "Usage: facial_capture -u <user> [options]\n\n"
    "Options:\n"
    "  -u, --user <name>          Username for which images will be captured\n"
    "  -c, --config <file>        Configuration file path\n"
    "                             (default: " FACIALAUTH_DEFAULT_CONFIG ")\n"
    "  -d, --device <path>        Video device to use (e.g. /dev/video0)\n"
    "  -w, --width <px>           Frame width\n"
    "  -h, --height <px>          Frame height\n"
    "  -f, --force                Force overwrite and restart numbering from img_1\n"
    "      --flush, --clean       Delete all stored images for the user\n"
    "  -n, --num-images <num>     Number of frames to capture\n"
    "  -s, --sleep <sec>          Delay between captures (seconds)\n"
    "      --detector <name>      Detection backend to use\n"
    "                             Available detectors:\n"
    "                                 haar          - Haar cascade model\n"
    "                                 yunet_fp32    - YuNet FP32 DNN model\n"
    "                                 yunet_int8    - YuNet INT8 optimized model\n"
    "      --list-detectors       List configured detectors\n"
    "      --list-devices         List V4L2 capture devices\n"
    "      --list-resolutions     List supported resolutions for the webcam\n"
    "      --format <fmt>         Output format: jpg | png | bmp\n"
    "  -v, --verbose              Print informational output\n"
    "      --debug                Print detailed debugging output\n"
    "      --nogui                Disable window previews\n"
    "  -H, --help                 Show help and exit\n";
}

static bool has_video_capture_capability(int fd)
{
    struct v4l2_capability cap{};
    if (ioctl(fd, VIDIOC_QUERYCAP, &cap) != 0)
        return false;

    return (cap.capabilities & V4L2_CAP_VIDEO_CAPTURE) ||
    (cap.capabilities & V4L2_CAP_VIDEO_CAPTURE_MPLANE);
}

static std::string read_sysfs_attr(const std::string &p)
{
    std::ifstream f(p);
    if (!f.is_open()) return {};
    std::string v; f >> v;
    return v;
}

static void list_devices_v4l()
{
    for (int i = 0; i < 32; ++i) {
        std::string dev = "/dev/video" + std::to_string(i);
        int fd = ::open(dev.c_str(), O_RDONLY | O_NONBLOCK);
        if (fd < 0) continue;

        if (!has_video_capture_capability(fd)) {
            close(fd);
            continue;
        }

        struct v4l2_capability cap{};
        if (ioctl(fd, VIDIOC_QUERYCAP, &cap) == 0) {
            std::string vendor  = read_sysfs_attr("/sys/class/video4linux/video" + std::to_string(i) + "/device/idVendor");
            std::string product = read_sysfs_attr("/sys/class/video4linux/video" + std::to_string(i) + "/device/idProduct");

            std::cout << dev << " ";
            if (!vendor.empty() && !product.empty())
                std::cout << "(" << vendor << ":" << product << ") ";

            std::cout << cap.card << "\n";
        }
        close(fd);
    }
}

static void list_resolutions_for_device(const std::string &dev)
{
    int fd = ::open(dev.c_str(), O_RDONLY | O_NONBLOCK);
    if (fd < 0) {
        std::cerr << "[ERRORE] Impossibile aprire " << dev << "\n";
        return;
    }

    if (!has_video_capture_capability(fd)) {
        std::cerr << "[ERRORE] Il device non supporta la cattura video.\n";
        close(fd);
        return;
    }

    std::cout << "Risoluzioni per " << dev << ":\n";

    struct v4l2_fmtdesc fmt{}; fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    for (fmt.index = 0; ioctl(fd, VIDIOC_ENUM_FMT, &fmt) == 0; fmt.index++) {
        uint32_t pf = fmt.pixelformat;
        char fourcc[5] = {
            char(pf & 0xFF),
            char((pf>>8) & 0xFF),
            char((pf>>16)&0xFF),
            char((pf>>24)&0xFF),
            0
        };

        std::cout << "  [" << fmt.index << "] " << fourcc << "\n";

        struct v4l2_frmsizeenum fr{};
        fr.pixel_format = pf;
        for (fr.index = 0; ioctl(fd, VIDIOC_ENUM_FRAMESIZES, &fr) == 0; fr.index++) {
            if (fr.type == V4L2_FRMSIZE_TYPE_DISCRETE)
                std::cout << "    " << fr.discrete.width << "x" << fr.discrete.height << "\n";
        }
    }

    close(fd);
}

static void debug_dump(const FacialAuthConfig &cfg)
{
    std::cerr << "[DEBUG] Config:\n"
    << " device="  << cfg.device  << "\n"
    << " width="   << cfg.width   << "\n"
    << " height="  << cfg.height  << "\n"
    << " frames="  << cfg.frames  << "\n"
    << " sleep_ms=" << cfg.sleep_ms << "\n"
    << " detector=" << cfg.detector_profile << "\n"
    << " format="   << cfg.image_format << "\n"
    << " debug="    << (cfg.debug?"yes":"no") << "\n"
    << " verbose="  << (cfg.verbose?"yes":"no") << "\n";
}

int facial_capture_main(int argc, char **argv)
{
    std::string user;
    std::string config_path = FACIALAUTH_DEFAULT_CONFIG;

    std::string device_override;
    int width_override = -1, height_override = -1, frames_override = -1;
    int sleep_sec = -1;

    std::string detector_override;
    std::string format_override;

    bool force=false, flush_only=false;
    bool verbose=false, debug=false, nogui=false;
    bool list_dev=false, list_res=false, list_det=false;

    for (int i = 1; i < argc; i++)
    {
        std::string a = argv[i];
        auto take = [&](const std::string &opt)->std::string{
            if (i+1>=argc) { std::cerr<<"Valore mancante per "<<opt<<"\n"; exit(1);}
            return argv[++i];
        };

        if (a=="-u"||a=="--user") user=take(a);
        else if (a=="-c"||a=="--config") config_path=take(a);
        else if (a=="-d"||a=="--device") device_override=take(a);
        else if (a=="-w"||a=="--width") width_override=std::stoi(take(a));
        else if (a=="-h"||a=="--height") height_override=std::stoi(take(a));
        else if (a=="-f"||a=="--force") force=true;
        else if (a=="--flush"||a=="--clean") flush_only=true;
        else if (a=="-n"||a=="--num-images") frames_override=std::stoi(take(a));
        else if (a=="-s"||a=="--sleep") sleep_sec=std::stoi(take(a));
        else if (a=="--detector") detector_override=take(a);
        else if (a=="--list-detectors") list_det=true;
        else if (a=="--list-devices") list_dev=true;
        else if (a=="--list-resolutions") list_res=true;
        else if (a=="--format") format_override=take(a);
        else if (a=="-v"||a=="--verbose") verbose=true;
        else if (a=="--debug") debug=true;
        else if (a=="--nogui") nogui=true;
        else if (a=="--help"||a=="-H") { print_help(); return 0; }
        else {
            std::cerr<<"Opzione sconosciuta "<<a<<"\n"; return 1;
        }
    }

    if (list_dev) { list_devices_v4l(); return 0; }

    FacialAuthConfig cfg;
    std::string log;

    if (!fa_load_config(cfg, log, config_path))
        std::cerr << log;

    cfg.debug |= debug;
    cfg.verbose |= verbose;

    if (!device_override.empty()) cfg.device=device_override;
    if (width_override>0) cfg.width=width_override;
    if (height_override>0) cfg.height=height_override;
    if (frames_override>0) cfg.frames=frames_override;
    if (sleep_sec>=0) cfg.sleep_ms=sleep_sec*1000;
    if (!format_override.empty()) cfg.image_format=format_override;

    if (force) cfg.force_overwrite=true;
    if (nogui) cfg.nogui=true;

    std::string fmt = cfg.image_format;
    std::transform(fmt.begin(), fmt.end(), fmt.begin(), ::tolower);
    if (fmt=="jpeg") fmt="jpg";
    if (fmt!="jpg" && fmt!="png" && fmt!="bmp") {
        std::cerr<<"[ERRORE] Formato non valido\n"; return 1;
    }
    cfg.image_format=fmt;

    if (list_det) {
        std::cout << "Detector configurati:\n";

        bool found = false;

        if (!cfg.haar_cascade_path.empty() && fa_file_exists(cfg.haar_cascade_path)) {
            std::cout << "  haar   -> " << cfg.haar_cascade_path << "\n";
            found = true;
        }

        if (!cfg.yunet_model.empty() && fa_file_exists(cfg.yunet_model)) {
            std::cout << "  yunet_fp32  -> " << cfg.yunet_model << "\n";
            found = true;
        }

        if (!cfg.yunet_model_int8.empty() && fa_file_exists(cfg.yunet_model_int8)) {
            std::cout << "  yunet_int8  -> " << cfg.yunet_model_int8 << "\n";
            found = true;
        }

        if (!found) {
            std::cout << "  (nessun detector rilevato nel config)\n";
        }

        return 0;
    }

    if (list_res) {
        list_resolutions_for_device(cfg.device);
        return 0;
    }

    if (flush_only) {
        if (user.empty()) {
            std::cerr<<"--flush richiede --user\n"; return 1;
        }
        std::string dir = fa_user_image_dir(cfg, user);
        namespace fs = std::filesystem;
        if (fs::exists(dir)) {
            for (auto &e : fs::directory_iterator(dir))
                fs::remove(e.path());
            std::cout<<"[INFO] Immagini cancellate\n";
        }
        return 0;
    }

    if (user.empty()) {
        std::cerr<<"Devi specificare --user\n"; return 1;
    }

    if (!detector_override.empty()) {

        std::string low = detector_override;
        std::transform(low.begin(), low.end(), low.begin(), ::tolower);

        bool valid = false;

        if (low == "haar" &&
            !cfg.haar_cascade_path.empty() &&
            fa_file_exists(cfg.haar_cascade_path))
        {
            valid = true;
        }
        else if ((low == "yunet" || low == "yunet_fp32") &&
            !cfg.yunet_model.empty() &&
            fa_file_exists(cfg.yunet_model))
        {
            valid = true;
            low = "yunet_fp32"; // normalizza
        }
        else if (low == "yunet_int8" &&
            !cfg.yunet_model_int8.empty() &&
            fa_file_exists(cfg.yunet_model_int8))
        {
            valid = true;
        }

        if (!valid) {
            std::cerr << "[ERRORE] Detector '" << detector_override
            << "' non definito nel file di configurazione\n";
            return 1;
        }

        cfg.detector_profile = low;
        if (cfg.debug)
            debug_dump(cfg);

    }

    std::string tool = "facial_capture";
    if (!fa_check_root(tool)) {
        std::cerr<<"Devi essere root\n"; return 1;
    }

    if (cfg.verbose)
        std::cerr<<"[INFO] Cattura avviata...\n";

    std::string capture_log;
    bool ok = fa_capture_images(user, cfg, cfg.image_format, capture_log);

    if (!ok) {
        std::cerr << capture_log;
        return 1;
    }

    std::cout << capture_log;

    if (cfg.verbose)
        std::cerr<<"[INFO] Cattura terminata.\n";

    return 0;
}

int main(int argc, char **argv)
{
    try {
        return facial_capture_main(argc, argv);
    }
    catch (const cv::Exception &e) {
        std::cerr << "[OpenCV] " << e.what() << "\n";
        return 1;
    }
    catch (const std::exception &e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }
}
