#include "../include/libfacialauth.h"

#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <unistd.h>

static void print_help()
{
    std::cout <<
    "Usage: facial_capture -u USER [options]\n"
    "\n"
    "General options:\n"
    "  -u, --user USER        Username (required)\n"
    "  -c, --config FILE      Config file path (default: "
    FACIALAUTH_CONFIG_DEFAULT ")\n"
    "  -d, --device DEV       Override video device (e.g. /dev/video0)\n"
    "  -w, --width N          Override capture width\n"
    "  -h, --height N         Override capture height\n"
    "  -n, --frames N         Override number of frames to capture\n"
    "  -s, --sleep MS         Delay between frames\n"
    "  -f, --force            Overwrite existing images\n"
    "  -g, --nogui            Disable GUI\n"
    "  -v, --debug            Enable debug logging\n"
    "      --detector NAME    auto|haar|yunet_fp32|yunet_int8\n"
    "      --clean            Remove user images\n"
    "      --reset            Remove user model + images\n"
    "      --format EXT       jpg|png\n"
    "\n"
    "Device inspection:\n"
    "      --list-devices             List available video devices\n"
    "      --list-resolutions DEV     List tested resolutions for /dev/videoX\n"
    "\n";
}

// Device/Resolution listing helpers

static int cmd_list_devices()
{
    std::vector<FaVideoDeviceInfo> devs;
    std::string logbuf;

    FacialAuthConfig dummy_cfg; // only for logging if needed

    if (!fa_list_video_devices(devs, logbuf)) {
        if (!logbuf.empty())
            std::cerr << logbuf;
        return 1;
    }

    for (const auto &d : devs) {
        std::cout << d.dev_node << "\n";

        if (!d.card.empty())
            std::cout << "  Card: " << d.card << "\n";
        if (!d.driver.empty())
            std::cout << "  Driver: " << d.driver << "\n";
        if (!d.bus_info.empty())
            std::cout << "  Bus: " << d.bus_info << "\n";

        if (!d.manufacturer.empty() || !d.product.empty()) {
            std::cout << "  Device: ";
            if (!d.manufacturer.empty())
                std::cout << d.manufacturer;
            if (!d.product.empty()) {
                if (!d.manufacturer.empty())
                    std::cout << " ";
                std::cout << d.product;
            }
            std::cout << "\n";
        }

        if (!d.usb_vendor_id.empty()) {
            std::cout << "  USB ID: " << normalize_hex4(d.usb_vendor_id)
            << ":" << normalize_hex4(d.usb_product_id) << "\n";
        }

        if (!d.pci_vendor_id.empty()) {
            std::cout << "  PCI ID: " << normalize_hex4(d.pci_vendor_id)
            << ":" << normalize_hex4(d.pci_device_id) << "\n";
        }

        std::cout << "\n";
    }

    if (!logbuf.empty())
        std::cerr << logbuf;

    return 0;
}

static int cmd_list_resolutions(const std::string &dev_node)
{
    std::vector<std::pair<int,int>> res;
    std::string logbuf;

    if (!fa_list_device_resolutions(dev_node, res, logbuf)) {
        if (!logbuf.empty())
            std::cerr << logbuf;
        return 1;
    }

    std::cout << "Supported (tested) resolutions for " << dev_node << ":\n";
    for (auto &r : res) {
        std::cout << "  " << r.first << "x" << r.second << "\n";
    }
    if (!logbuf.empty())
        std::cerr << logbuf;

    return 0;
}

// Main capture wrapper

int facial_capture_main(int argc, char *argv[])
{
    const char *prog = "facial_capture";
    if (!fa_check_root(prog))
        return 1;

    std::string user;
    std::string cfg_path;
    std::string opt_format;

    bool opt_force  = false;
    bool opt_clean  = false;
    bool opt_reset  = false;
    bool opt_debug  = false;
    bool opt_nogui  = false;

    bool opt_list_devices     = false;
    bool opt_list_resolutions = false;
    std::string list_res_dev;

    std::string opt_device;
    std::string opt_detector;

    int opt_width  = -1;
    int opt_height = -1;
    int opt_frames = -1;
    int opt_sleep  = -1;

    // Parse CLI
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];

        if (a == "--list-devices") {
            opt_list_devices = true;
        }
        else if (a == "--list-resolutions" && i + 1 < argc) {
            opt_list_resolutions = true;
            list_res_dev = argv[++i];
        }
        else if ((a == "-u" || a == "--user") && i + 1 < argc)
            user = argv[++i];
        else if ((a == "-c" || a == "--config") && i + 1 < argc)
            cfg_path = argv[++i];
        else if ((a == "-d" || a == "--device") && i + 1 < argc)
            opt_device = argv[++i];
        else if ((a == "-w" || a == "--width") && i + 1 < argc)
            opt_width = std::atoi(argv[++i]);
        else if ((a == "-h" || a == "--height") && i + 1 < argc)
            opt_height = std::atoi(argv[++i]);
        else if ((a == "-n" || a == "--frames") && i + 1 < argc)
            opt_frames = std::atoi(argv[++i]);
        else if ((a == "-s" || a == "--sleep") && i + 1 < argc)
            opt_sleep = std::atoi(argv[++i]);
        else if (a == "-f" || a == "--force")
            opt_force = true;
        else if (a == "-g" || a == "--nogui")
            opt_nogui = true;
        else if (a == "-v" || a == "--debug")
            opt_debug = true;
        else if (a == "--detector" && i + 1 < argc)
            opt_detector = argv[++i];
        else if (a == "--clean")
            opt_clean = true;
        else if (a == "--reset")
            opt_reset = true;
        else if (a == "--format" && i + 1 < argc)
            opt_format = argv[++i];
        else if (a == "--help") {
            print_help();
            return 0;
        }
    }

    // Device/resolution-only commands do not require user/config
    if (opt_list_devices) {
        return cmd_list_devices();
    }
    if (opt_list_resolutions) {
        if (list_res_dev.empty()) {
            std::cerr << "--list-resolutions requires a device path\n";
            return 1;
        }
        return cmd_list_resolutions(list_res_dev);
    }

    if (user.empty()) {
        print_help();
        return 1;
    }

    FacialAuthConfig cfg;
    std::string logbuf;

    fa_load_config(cfg, logbuf,
                   cfg_path.empty() ? FACIALAUTH_CONFIG_DEFAULT : cfg_path);

    if (!logbuf.empty())
        std::cerr << logbuf;
    logbuf.clear();

    // Apply CLI overrides
    if (!opt_device.empty())     cfg.device           = opt_device;
    if (!opt_detector.empty())   cfg.detector_profile = opt_detector;
    if (opt_width  > 0)          cfg.width            = opt_width;
    if (opt_height > 0)          cfg.height           = opt_height;
    if (opt_frames > 0)          cfg.frames           = opt_frames;
    if (opt_sleep >= 0)          cfg.sleep_ms         = opt_sleep;
    if (opt_debug)               cfg.debug            = true;
    if (opt_nogui)               cfg.nogui            = true;
    if (!opt_format.empty())     cfg.image_format     = opt_format;

    std::string user_img_dir = fa_user_image_dir(cfg, user);
    std::string user_model   = fa_user_model_path(cfg, user);

    if (opt_reset) {
        if (!user_img_dir.empty())
            fs::remove_all(user_img_dir);
        if (fs::exists(user_model))
            fs::remove(user_model);
        return 0;
    }

    if (opt_clean) {
        if (!user_img_dir.empty())
            fs::remove_all(user_img_dir);
        return 0;
    }

    if (opt_force && !user_img_dir.empty())
        fs::remove_all(user_img_dir);

    bool ok = fa_capture_images(user, cfg, cfg.image_format, logbuf);

    if (!logbuf.empty())
        std::cerr << logbuf;

    return ok ? 0 : 1;
}

int main(int argc, char *argv[])
{
    return facial_capture_main(argc, argv);
}
