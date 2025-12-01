#include "../include/libfacialauth.h"

#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <unistd.h>
#include <filesystem>
#include <fstream>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>
#include <sstream>
#include <iomanip>

namespace fs = std::filesystem;

// =============================================================
//  Helpers
// =============================================================

// hex formatter for USB/Pci IDs
static std::string normalize_hex4(int value) {
    std::ostringstream oss;
    oss << std::hex << std::setw(4) << std::setfill('0')
    << std::nouppercase << (value & 0xffff);
    return oss.str();
}

// Read file to string (safe)
static std::string read_file_safe(const std::string &path) {
    std::ifstream f(path);
    if (!f.good()) return "";
    std::string s;
    std::getline(f, s);
    return s;
}

static std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim))
        out.push_back(item);
    return out;
}


// =============================================================
//  Decode vendor/product from usb.ids / pci.ids (STRICT MODE)
// =============================================================

static const std::vector<std::string> IDS_PATHS_USB = {
    "/usr/share/hwdata/usb.ids",
    "/usr/share/misc/usb.ids",
    "/usr/local/share/hwdata/usb.ids",
    "/usr/local/share/misc/usb.ids"
};

static const std::vector<std::string> IDS_PATHS_PCI = {
    "/usr/share/hwdata/pci.ids",
    "/usr/share/misc/pci.ids",
    "/usr/local/share/hwdata/pci.ids",
    "/usr/local/share/misc/pci.ids"
};

static std::string decode_usb_vendor(const std::string &vendor_hex) {
    for (auto &p : IDS_PATHS_USB) {
        std::ifstream f(p);
        if (!f.good()) continue;

        std::string line;
        while (std::getline(f, line)) {
            if (line.size() > 4 && line[0] != '\t' &&
                line.substr(0,4) == vendor_hex)
                return line.substr(5);
        }
    }
    std::cerr << "[WARN] USB vendor " << vendor_hex
    << " not found in usb.ids\n";
    return "";
}

static std::string decode_usb_product(const std::string &vendor_hex,
                                      const std::string &product_hex) {
    for (auto &p : IDS_PATHS_USB) {
        std::ifstream f(p);
        if (!f.good()) continue;

        std::string line;
        bool in_vendor = false;

        while (std::getline(f, line)) {
            if (line.size() > 4 && line[0] != '\t') {
                in_vendor = (line.substr(0,4) == vendor_hex);
            } else if (in_vendor && line.size() > 5 && line[1] != '\t') {
                if (line.substr(1,4) == product_hex)
                    return line.substr(6);
            }
        }
    }
    std::cerr << "[WARN] USB product " << vendor_hex << ":" << product_hex
    << " not found in usb.ids\n";
    return "";
                                      }

                                      static std::string decode_pci_vendor(const std::string &vendor_hex) {
                                          for (auto &p : IDS_PATHS_PCI) {
                                              std::ifstream f(p);
                                              if (!f.good()) continue;

                                              std::string line;
                                              while (std::getline(f, line)) {
                                                  if (line.size() > 4 && line[0] != '\t' &&
                                                      line.substr(0,4) == vendor_hex)
                                                      return line.substr(5);
                                              }
                                          }
                                          std::cerr << "[WARN] PCI vendor " << vendor_hex
                                          << " not found in pci.ids\n";
                                          return "";
                                      }

                                      static std::string decode_pci_product(const std::string &vendor_hex,
                                                                            const std::string &product_hex) {
                                          for (auto &p : IDS_PATHS_PCI) {
                                              std::ifstream f(p);
                                              if (!f.good()) continue;

                                              std::string line;
                                              bool in_vendor = false;

                                              while (std::getline(f, line)) {
                                                  if (line.size() > 4 && line[0] != '\t') {
                                                      in_vendor = (line.substr(0,4) == vendor_hex);
                                                  } else if (in_vendor && line.size() > 5 && line[1] != '\t') {
                                                      if (line.substr(1,4) == product_hex)
                                                          return line.substr(6);
                                                  }
                                              }
                                          }
                                          std::cerr << "[WARN] PCI product " << vendor_hex << ":" << product_hex
                                          << " not found in pci.ids\n";
                                          return "";
                                                                            }


                                                                            // =============================================================
                                                                            //  List available /dev/video* devices with metadata
                                                                            // =============================================================

                                                                            static void list_devices() {
                                                                                std::vector<std::string> devs;

                                                                                for (int i = 0; i < 10; i++) {
                                                                                    std::string dev = "/dev/video" + std::to_string(i);
                                                                                    if (access(dev.c_str(), R_OK) == 0)
                                                                                        devs.push_back(dev);
                                                                                }

                                                                                if (devs.empty()) {
                                                                                    std::cout << "No video devices found.\n";
                                                                                    return;
                                                                                }

                                                                                for (auto &dev : devs) {
                                                                                    std::cout << dev << "\n";

                                                                                    int fd = open(dev.c_str(), O_RDWR);
                                                                                    if (fd < 0) {
                                                                                        std::cout << "  (unable to open)\n\n";
                                                                                        continue;
                                                                                    }

                                                                                    struct v4l2_capability cap {};
                                                                                    if (ioctl(fd, VIDIOC_QUERYCAP, &cap) < 0) {
                                                                                        std::cout << "  (querycap failed)\n\n";
                                                                                        close(fd);
                                                                                        continue;
                                                                                    }

                                                                                    std::string card = (const char*)cap.card;
                                                                                    std::string driver = (const char*)cap.driver;
                                                                                    std::string bus = (const char*)cap.bus_info;

                                                                                    std::cout << "  Name:   " << card << "\n";
                                                                                    std::cout << "  Driver: " << driver << "\n";
                                                                                    std::cout << "  Bus:    " << bus << "\n";

                                                                                    close(fd);

                                                                                    // sysfs
                                                                                    std::string sys = "/sys/class/video4linux/" +
                                                                                    fs::path(dev).filename().string() + "/device";

                                                                                    if (!fs::exists(sys)) {
                                                                                        std::cout << "  (no sysfs metadata available)\n\n";
                                                                                        continue;
                                                                                    }

                                                                                    // Follow symlink up to usb or pci device directory
                                                                                    std::string real = fs::canonical(sys);
                                                                                    std::string parent = fs::canonical(real + "/..");

                                                                                    std::string idVendor = read_file_safe(parent + "/idVendor");
                                                                                    std::string idProduct = read_file_safe(parent + "/idProduct");

                                                                                    if (!idVendor.empty() && !idProduct.empty()) {
                                                                                        idVendor = normalize_hex4(std::stoi(idVendor, nullptr, 16));
                                                                                        idProduct = normalize_hex4(std::stoi(idProduct, nullptr, 16));

                                                                                        std::cout << "  USB ID: " << idVendor << ":" << idProduct << "\n";

                                                                                        std::string vname = decode_usb_vendor(idVendor);
                                                                                        std::string pname = decode_usb_product(idVendor, idProduct);
                                                                                        if (!vname.empty()) std::cout << "  Vendor: " << vname << "\n";
                                                                                        if (!pname.empty()) std::cout << "  Model:  " << pname << "\n";
                                                                                    }

                                                                                    std::cout << "\n";
                                                                                }
                                                                            }


                                                                            // =============================================================
                                                                            //  Test common resolutions (V4L2 + Fallback OpenCV)
                                                                            // =============================================================

                                                                            static const std::vector<std::pair<int,int>> test_res = {
                                                                                {320,240}, {640,480}, {800,600},
                                                                                {1024,768}, {1280,720}, {1920,1080}
                                                                            };

                                                                            static void list_resolutions(const std::string &dev) {
                                                                                std::cout << "Supported (tested) resolutions for " << dev << ":\n";

                                                                                int fd = open(dev.c_str(), O_RDWR);
                                                                                if (fd < 0) {
                                                                                    std::cout << "  (unable to open device)\n";
                                                                                    return;
                                                                                }

                                                                                bool found_any = false;

                                                                                for (auto &r : test_res) {
                                                                                    struct v4l2_format fmt {};
                                                                                    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                                                                                    fmt.fmt.pix.width = r.first;
                                                                                    fmt.fmt.pix.height = r.second;
                                                                                    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;

                                                                                    if (ioctl(fd, VIDIOC_TRY_FMT, &fmt) == 0) {
                                                                                        if (fmt.fmt.pix.width == r.first &&
                                                                                            fmt.fmt.pix.height == r.second) {
                                                                                            std::cout << "  " << r.first << "x" << r.second << "\n";
                                                                                        found_any = true;
                                                                                            }
                                                                                    }
                                                                                }

                                                                                close(fd);

                                                                                if (!found_any)
                                                                                    std::cout << "  (no standard resolutions detected)\n";
                                                                            }


                                                                            // =============================================================
                                                                            //  HELP
                                                                            // =============================================================

                                                                            static void print_help()
                                                                            {
                                                                                std::cout <<
                                                                                "Usage: facial_capture -u USER [options]\n"
                                                                                "  -u, --user USER        Username\n"
                                                                                "  -d, --device DEV       Override device\n"
                                                                                "  -w, --width N          Override width\n"
                                                                                "  -h, --height N         Override height\n"
                                                                                "  -n, --frames N         Override number of frames\n"
                                                                                "  -s, --sleep MS         Delay between frames\n"
                                                                                "  -f, --force            Overwrite existing images\n"
                                                                                "  -g, --nogui            Disable GUI\n"
                                                                                "      --detector NAME    auto|haar|yunet|yunet_int8\n"
                                                                                "      --clean            Remove user images\n"
                                                                                "      --reset            Remove user model + images\n"
                                                                                "      --format EXT       jpg|png\n"
                                                                                "  -v, --debug            Enable debug\n"
                                                                                "  -c, --config FILE      Config file path\n"
                                                                                "      --list-devices     List video devices\n"
                                                                                "      --list-resolutions DEV   List resolution support\n"
                                                                                "\n";
                                                                            }



                                                                            // =============================================================
                                                                            //  MAIN WRAPPER
                                                                            // =============================================================

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

                                                                                std::string opt_device;
                                                                                std::string opt_detector;

                                                                                int opt_width  = -1;
                                                                                int opt_height = -1;
                                                                                int opt_frames = -1;
                                                                                int opt_sleep  = -1;

                                                                                // ------------------------------------------------------------
                                                                                // Parse arguments
                                                                                // ------------------------------------------------------------
                                                                                for (int i = 1; i < argc; ++i) {
                                                                                    std::string a = argv[i];

                                                                                    if (a == "--list-devices") {
                                                                                        list_devices();
                                                                                        return 0;
                                                                                    }
                                                                                    if (a == "--list-resolutions" && i + 1 < argc) {
                                                                                        list_resolutions(argv[++i]);
                                                                                        return 0;
                                                                                    }

                                                                                    if ((a == "-u" || a == "--user") && i + 1 < argc)
                                                                                        user = argv[++i];
                                                                                    else if ((a == "-c" || a == "--config") && i + 1 < argc)
                                                                                        cfg_path = argv[++i];
                                                                                    else if ((a == "-d" || a == "--device") && i + 1 < argc)
                                                                                        opt_device = argv[++i];
                                                                                    else if ((a == "-w" || a == "--width") && i + 1 < argc)
                                                                                        opt_width = atoi(argv[++i]);
                                                                                    else if ((a == "-h" || a == "--height") && i + 1 < argc)
                                                                                        opt_height = atoi(argv[++i]);
                                                                                    else if ((a == "-n" || a == "--frames") && i + 1 < argc)
                                                                                        opt_frames = atoi(argv[++i]);
                                                                                    else if ((a == "-s" || a == "--sleep") && i + 1 < argc)
                                                                                        opt_sleep = atoi(argv[++i]);
                                                                                    else if (a == "-f" || a == "--force")
                                                                                        opt_force = true;
                                                                                    else if (a == "-g" || a == "--nogui")
                                                                                        opt_nogui = true;
                                                                                    else if (a == "-v" || a == "-Opencv-v-on-source=++debug")
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

                                                                                if (user.empty()) {
                                                                                    print_help();
                                                                                    return 1;
                                                                                }

                                                                                // ------------------------------------------------------------
                                                                                // Load config
                                                                                // ------------------------------------------------------------
                                                                                FacialAuthConfig cfg;
                                                                                std::string logbuf;

                                                                                fa_load_config(cfg, logbuf,
                                                                                               cfg_path.empty() ? FACIALAUTH_CONFIG_DEFAULT : cfg_path);

                                                                                if (!logbuf.empty())
                                                                                    std::cerr << logbuf;
                                                                                logbuf.clear();

                                                                                // CLI overrides
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

                                                                                // ------------------------------------------------------------
                                                                                // Reset (remove images + model)
                                                                                // ------------------------------------------------------------
                                                                                if (opt_reset) {
                                                                                    bool removed = false;

                                                                                    if (fs::exists(user_img_dir)) {
                                                                                        fs::remove_all(user_img_dir);
                                                                                        std::cout << "[INFO] Removed all images for '" << user << "'\n";
                                                                                        removed = true;
                                                                                    }

                                                                                    if (fs::exists(user_model)) {
                                                                                        fs::remove(user_model);
                                                                                        std::cout << "[INFO] Removed model for '" << user << "'\n";
                                                                                        removed = true;
                                                                                    }

                                                                                    if (!removed)
                                                                                        std::cout << "[INFO] Nothing to reset for '" << user << "'\n";

                                                                                    return 0;
                                                                                }

                                                                                // ------------------------------------------------------------
                                                                                // Clean images
                                                                                // ------------------------------------------------------------
                                                                                if (opt_clean) {
                                                                                    if (fs::exists(user_img_dir)) {
                                                                                        fs::remove_all(user_img_dir);
                                                                                        std::cout << "[INFO] Removed images for '" << user << "'\n";
                                                                                    } else {
                                                                                        std::cout << "[INFO] No images to remove\n";
                                                                                    }
                                                                                    return 0;
                                                                                }

                                                                                // ------------------------------------------------------------
                                                                                // Force recreate image directory
                                                                                // ------------------------------------------------------------
                                                                                if (opt_force) {
                                                                                    if (fs::exists(user_img_dir)) {
                                                                                        fs::remove_all(user_img_dir);
                                                                                        std::cout << "[INFO] Forced removal of old images\n";
                                                                                    }
                                                                                }

                                                                                // ------------------------------------------------------------
                                                                                // Perform capture
                                                                                // ------------------------------------------------------------
                                                                                bool ok = fa_capture_images(user, cfg, cfg.image_format, logbuf);

                                                                                if (!logbuf.empty())
                                                                                    std::cerr << logbuf;

                                                                                return ok ? 0 : 1;
                                                                            }

                                                                            int main(int argc, char *argv[])
                                                                            {
                                                                                return facial_capture_main(argc, argv);
                                                                            }
