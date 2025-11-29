#include "libfacialauth.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/face.hpp>

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <thread>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace cv;
using namespace std;

// ------------------------------------------------------------
// small helpers
// ------------------------------------------------------------

static inline string trim(const string &s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == string::npos) return "";
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

static inline bool str_to_bool(const string &v, bool defv) {
    string s;
    for (char c : v) s.push_back(::tolower(c));
    if (s == "1" || s == "true" || s == "yes" || s == "on") return true;
    if (s == "0" || s == "false" || s == "no" || s == "off") return false;
    return defv;
}

static inline string join_path(const string &a, const string &b) {
    if (a.empty()) return b;
    if (a.back() == '/') return a + b;
    return a + "/" + b;
}

bool fa_file_exists(const string &path) {
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

bool fa_ensure_dir(const string &path) {
    if (path.empty()) return false;
    struct stat st{};
    if (::stat(path.c_str(), &st) == 0) {
        return S_ISDIR(st.st_mode);
    }
    // simple mkdir -p for single level: we assume parent exists
    if (::mkdir(path.c_str(), 0755) == 0) return true;
    // if parent does not exist, try recursing
    auto pos = path.find_last_of('/');
    if (pos != string::npos) {
        string parent = path.substr(0, pos);
        if (!parent.empty() && !fa_ensure_dir(parent)) return false;
    }
    return ::mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
}

void fa_msleep(int ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

// ------------------------------------------------------------
// logging
// ------------------------------------------------------------

static void log_to_file(const FacialAuthConfig &cfg,
                        const string &level,
                        const string &msg) {
    if (cfg.log_file.empty()) return;
    ofstream ofs(cfg.log_file, ios::app);
    if (!ofs) return;
    ofs << "[" << level << "] " << msg << "\n";
}

static void log_debug(const FacialAuthConfig &cfg, const string &msg) {
    if (!cfg.debug) return;
    log_to_file(cfg, "DEBUG", msg);
}

static void log_info(const FacialAuthConfig &cfg, const string &msg) {
    log_to_file(cfg, "INFO", msg);
}

static void log_error(const FacialAuthConfig &cfg, const string &msg) {
    log_to_file(cfg, "ERROR", msg);
}

// ------------------------------------------------------------
// config
// ------------------------------------------------------------

bool fa_read_config(const string &path, FacialAuthConfig &cfg, string &err) {
    if (!fa_file_exists(path)) {
        err = "Config file not found: " + path;
        return false;
    }

    ifstream ifs(path);
    if (!ifs) {
        err = "Cannot open config file: " + path;
        return false;
    }

    string line;
    int lineno = 0;
    while (std::getline(ifs, line)) {
        ++lineno;
        line = trim(line);
        if (line.empty() || line[0] == '#')
            continue;

        auto eq = line.find('=');
        if (eq == string::npos) continue;

        string key = trim(line.substr(0, eq));
        string val = trim(line.substr(eq + 1));

        if (key == "basedir") {
            cfg.basedir = val;
        } else if (key == "device") {
            cfg.device = val;
        } else if (key == "fallback_device") {
            cfg.fallback_device = str_to_bool(val, true);
        } else if (key == "width") {
            cfg.width = std::atoi(val.c_str());
        } else if (key == "height") {
            cfg.height = std::atoi(val.c_str());
        } else if (key == "capture_count") {
            cfg.capture_count = std::atoi(val.c_str());
        } else if (key == "capture_delay") {
            cfg.capture_delay = std::atoi(val.c_str());
        } else if (key == "detector_profile") {
            cfg.detector_profile = val;
        } else if (key == "recognizer") {
            cfg.recognizer = val;
        } else if (key == "haar_model") {
            cfg.haar_cascade_path = val;
        } else if (key == "yunet_model") {
            cfg.yunet_model = val;
        } else if (key == "yunet_model_int8") {
            cfg.yunet_model_int8 = val;
        } else if (key == "sface_model") {
            cfg.sface_model = val;
        } else if (key == "sface_model_int8") {
            cfg.sface_model_int8 = val;
        } else if (key == "eigen_threshold") {
            cfg.eigen_threshold = atof(val.c_str());
        } else if (key == "fisher_threshold") {
            cfg.fisher_threshold = atof(val.c_str());
        } else if (key == "lbph_threshold") {
            cfg.lbph_threshold = atof(val.c_str());
        } else if (key == "sface_threshold") {
            cfg.sface_threshold = atof(val.c_str());
        } else if (key == "debug") {
            cfg.debug = str_to_bool(val, false);
        } else if (key == "save_failed_images") {
            cfg.save_failed_images = str_to_bool(val, false);
        } else if (key == "log_file") {
            cfg.log_file = val;
        } else if (key == "force_overwrite") {
            cfg.force_overwrite = str_to_bool(val, false);
        } else if (key == "dnn_backend") {
            cfg.dnn_backend = val;
        } else {
            // ignore unknown key
            std::ostringstream oss;
            oss << "Unknown key in config line " << lineno << ": " << key;
            log_debug(cfg, oss.str());
        }
    }

    // normalize
    if (cfg.capture_count <= 0) cfg.capture_count = 50;
    if (cfg.capture_delay <= 0) cfg.capture_delay = 50;
    if (cfg.width <= 0) cfg.width = 640;
    if (cfg.height <= 0) cfg.height = 480;

    return true;
}

// ------------------------------------------------------------
// path helpers
// ------------------------------------------------------------

string fa_user_image_dir(const FacialAuthConfig &cfg, const string &user) {
    return join_path(join_path(cfg.basedir, "images"), user);
}

string fa_user_model_path(const FacialAuthConfig &cfg, const string &user) {
    return join_path(join_path(cfg.basedir, "models"), user + ".xml");
}

// ------------------------------------------------------------
// detector creation
// ------------------------------------------------------------

struct DetectorWrapper {
    enum Kind { NONE, HAAR, YUNET } kind = NONE;
    CascadeClassifier haar;
    Ptr<FaceDetectorYN> yunet;
};

static DetectorWrapper create_detector(const FacialAuthConfig &cfg) {
    DetectorWrapper w;

    string prof = cfg.detector_profile;
    for (auto &c : prof) c = ::tolower(c);

    bool want_auto = (prof == "auto");
    bool want_yunet = (prof == "yunet" || prof == "yunet_int8");
    bool want_haar = (prof == "haar");

    // decide what we'll actually use
    if (want_auto) {
        if (!cfg.yunet_model.empty() && fa_file_exists(cfg.yunet_model))
            want_yunet = true;
        else
            want_haar = true;
    }

    if (want_yunet && !cfg.yunet_model.empty() && fa_file_exists(cfg.yunet_model)) {
        // backend/target are currently only logged; FaceDetectorYN internally
        // uses DNN and OpenCV build options.
        int backend = cv::dnn::DNN_BACKEND_DEFAULT;
        int target = cv::dnn::DNN_TARGET_CPU;

        string be = cfg.dnn_backend;
        for (auto &c : be) c = ::tolower(c);
        if (be == "cuda") {
            backend = cv::dnn::DNN_BACKEND_CUDA;
            target = cv::dnn::DNN_TARGET_CUDA;
        }

        w.yunet = FaceDetectorYN::create(
            cfg.yunet_model,
            "",
            Size(cfg.width, cfg.height),
            0.9f,
            0.3f,
            5000,
            backend,
            target
        );
        if (!w.yunet.empty()) {
            w.kind = DetectorWrapper::YUNET;
            std::ostringstream oss;
            oss << "Using detector: YUNET model=" << cfg.yunet_model
                << " backend=" << cfg.dnn_backend;
            log_info(cfg, oss.str());
            return w;
        } else {
            log_error(cfg, "Failed to create YUNET detector, falling back to HAAR if available");
        }
    }

    if (want_haar && !cfg.haar_cascade_path.empty() && fa_file_exists(cfg.haar_cascade_path)) {
        if (w.haar.load(cfg.haar_cascade_path)) {
            w.kind = DetectorWrapper::HAAR;
            std::ostringstream oss;
            oss << "Using detector: HAAR model=" << cfg.haar_cascade_path;
            log_info(cfg, oss.str());
            return w;
        } else {
            log_error(cfg, "Failed to load HAAR cascade: " + cfg.haar_cascade_path);
        }
    }

    log_error(cfg, "No valid detector available (profile=" + cfg.detector_profile + ")");
    return w;
}

// ------------------------------------------------------------
// capture
// ------------------------------------------------------------

bool fa_capture_images(const FacialAuthConfig &cfg,
                       const string &user,
                       int max_images,
                       string &err) {
    FacialAuthConfig local_cfg = cfg;
    if (max_images > 0) local_cfg.capture_count = max_images;

    string img_dir = fa_user_image_dir(local_cfg, user);
    if (!fa_ensure_dir(img_dir)) {
        err = "Cannot create image dir: " + img_dir;
        return false;
    }

    VideoCapture cap;
    if (!cap.open(local_cfg.device)) {
        if (local_cfg.fallback_device) {
            log_error(local_cfg, "Failed to open device " + local_cfg.device + ", trying /dev/video1");
            if (!cap.open("/dev/video1")) {
                err = "Cannot open camera device(s)";
                return false;
            }
        } else {
            err = "Cannot open camera device: " + local_cfg.device;
            return false;
        }
    }

    cap.set(CAP_PROP_FRAME_WIDTH, local_cfg.width);
    cap.set(CAP_PROP_FRAME_HEIGHT, local_cfg.height);

    DetectorWrapper det = create_detector(local_cfg);
    if (det.kind == DetectorWrapper::NONE) {
        err = "No detector available; check configuration";
        return false;
    }

    log_info(local_cfg, "Starting capture for user=" + user);

    int saved = 0;
    Mat frame;

    while (saved < local_cfg.capture_count) {
        if (!cap.read(frame) || frame.empty()) {
            err = "Failed to read frame from camera";
            log_error(local_cfg, err);
            return false;
        }

        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        vector<Rect> faces;

        if (det.kind == DetectorWrapper::HAAR) {
            det.haar.detectMultiScale(gray, faces, 1.1, 3, 0, Size(80, 80));
        } else if (det.kind == DetectorWrapper::YUNET && !det.yunet.empty()) {
            Mat resized;
            if (frame.size() != Size(local_cfg.width, local_cfg.height))
                resize(frame, resized, Size(local_cfg.width, local_cfg.height));
            else
                resized = frame;

            Mat dets;
            det.yunet->detect(resized, dets);
            for (int i = 0; i < dets.rows; ++i) {
                float left   = dets.at<float>(i, 0);
                float top    = dets.at<float>(i, 1);
                float width  = dets.at<float>(i, 2);
                float height = dets.at<float>(i, 3);
                float score  = dets.at<float>(i, 4);
                if (score < 0.9f) continue;
                faces.emplace_back((int)left, (int)top, (int)width, (int)height);
            }
        }

        if (faces.empty()) {
            log_debug(local_cfg, "No faces detected in frame");
            fa_msleep(local_cfg.capture_delay);
            continue;
        }

        // take the largest face
        Rect best = faces[0];
        for (const auto &r : faces) {
            if (r.area() > best.area())
                best = r;
        }

        Mat face = gray(best).clone();
        resize(face, face, Size(200, 200));

        char namebuf[256];
        std::snprintf(namebuf, sizeof(namebuf), "img_%03d.jpg", saved + 1);
        string out_path = join_path(img_dir, namebuf);

        if (!imwrite(out_path, face)) {
            err = "Failed to save image: " + out_path;
            log_error(local_cfg, err);
            return false;
        }

        std::ostringstream oss;
        oss << "Saved " << out_path;
        log_info(local_cfg, oss.str());

        ++saved;
        fa_msleep(local_cfg.capture_delay);
    }

    log_info(local_cfg, "Capture completed for user=" + user);
    return true;
}

// ------------------------------------------------------------
// training (Eigen / Fisher / LBPH)
// ------------------------------------------------------------

static Ptr<face::FaceRecognizer> create_recognizer(const string &method,
                                                   const FacialAuthConfig &cfg) {
    string m = method;
    for (auto &c : m) c = ::tolower(c);
    if (m == "eigen") {
        auto r = face::EigenFaceRecognizer::create();
        r->setThreshold(cfg.eigen_threshold);
        return r;
    } else if (m == "fisher") {
        auto r = face::FisherFaceRecognizer::create();
        r->setThreshold(cfg.fisher_threshold);
        return r;
    } else { // default lbph
        auto r = face::LBPHFaceRecognizer::create();
        r->setThreshold(cfg.lbph_threshold);
        return r;
    }
}

bool fa_train_user(const FacialAuthConfig &cfg,
                   const string &user,
                   const string &method_in,
                   string &err) {
    string method = method_in;
    if (method.empty() || method == "auto")
        method = cfg.recognizer.empty() ? "lbph" : cfg.recognizer;

    string img_dir = fa_user_image_dir(cfg, user);
    string model_path = fa_user_model_path(cfg, user);

    if (!fa_ensure_dir(join_path(cfg.basedir, "models"))) {
        err = "Cannot create models directory under " + cfg.basedir;
        return false;
    }

    vector<string> image_files;

    // very small directory scan based on sequential file names img_XXX.jpg
    for (int i = 1; i <= 999; ++i) {
        char namebuf[256];
        std::snprintf(namebuf, sizeof(namebuf), "img_%03d.jpg", i);
        string path = join_path(img_dir, namebuf);
        if (!fa_file_exists(path)) continue;
        image_files.push_back(path);
    }

    if (image_files.empty()) {
        err = "No training images found in " + img_dir;
        return false;
    }

    vector<Mat> images;
    vector<int> labels;
    for (const auto &p : image_files) {
        Mat img = imread(p, IMREAD_GRAYSCALE);
        if (img.empty()) {
            log_error(cfg, "Failed to read training image: " + p);
            continue;
        }
        resize(img, img, Size(200, 200));
        images.push_back(img);
        labels.push_back(0); // single-user model
    }

    if (images.empty()) {
        err = "No valid training images for user " + user;
        return false;
    }

    Ptr<face::FaceRecognizer> rec = create_recognizer(method, cfg);
    if (rec.empty()) {
        err = "Could not create recognizer for method: " + method;
        return false;
    }

    log_info(cfg, "Training user " + user + " with method " + method);

    rec->train(images, labels);
    rec->save(model_path);

    log_info(cfg, "Model saved to " + model_path);
    return true;
}

// ------------------------------------------------------------
// testing
// ------------------------------------------------------------

bool fa_test_user(const FacialAuthConfig &cfg,
                  const string &user,
                  double &best_conf,
                  string &used_method,
                  string &err) {
    string model_path = fa_user_model_path(cfg, user);
    if (!fa_file_exists(model_path)) {
        err = "Model not found for user " + user + ": " + model_path;
        return false;
    }

    // detect recognizer type from XML header
    ifstream ifs(model_path);
    if (!ifs) {
        err = "Cannot open model XML: " + model_path;
        return false;
    }
    string first_lines, line;
    int cnt = 0;
    while (cnt < 20 && std::getline(ifs, line)) {
        first_lines += line;
        ++cnt;
    }
    ifs.close();

    string xml_lower = first_lines;
    for (auto &c : xml_lower) c = ::tolower(c);

    string method = "lbph";
    if (xml_lower.find("eigenfaces") != string::npos)
        method = "eigen";
    else if (xml_lower.find("fisherfaces") != string::npos)
        method = "fisher";

    Ptr<face::FaceRecognizer> rec = create_recognizer(method, cfg);
    if (rec.empty()) {
        err = "Could not create recognizer of detected type: " + method;
        return false;
    }

    rec->read(model_path);

    VideoCapture cap;
    if (!cap.open(cfg.device)) {
        if (cfg.fallback_device) {
            if (!cap.open("/dev/video1")) {
                err = "Cannot open camera device(s)";
                return false;
            }
        } else {
            err = "Cannot open camera device " + cfg.device;
            return false;
        }
    }
    cap.set(CAP_PROP_FRAME_WIDTH, cfg.width);
    cap.set(CAP_PROP_FRAME_HEIGHT, cfg.height);

    DetectorWrapper det = create_detector(cfg);
    if (det.kind == DetectorWrapper::NONE) {
        err = "No detector available";
        return false;
    }

    log_info(cfg, "Starting authentication for user=" + user + " method=" + method);

    Mat frame, gray;
    best_conf = 1e12;
    used_method = method;

    for (int i = 0; i < 20; ++i) {
        if (!cap.read(frame) || frame.empty()) {
            err = "Failed to read frame from camera";
            return false;
        }
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        vector<Rect> faces;
        if (det.kind == DetectorWrapper::HAAR) {
            det.haar.detectMultiScale(gray, faces, 1.1, 3, 0, Size(80, 80));
        } else if (det.kind == DetectorWrapper::YUNET && !det.yunet.empty()) {
            Mat resized;
            if (frame.size() != Size(cfg.width, cfg.height))
                resize(frame, resized, Size(cfg.width, cfg.height));
            else
                resized = frame;

            Mat dets;
            det.yunet->detect(resized, dets);
            for (int r = 0; r < dets.rows; ++r) {
                float left   = dets.at<float>(r, 0);
                float top    = dets.at<float>(r, 1);
                float width  = dets.at<float>(r, 2);
                float height = dets.at<float>(r, 3);
                float score  = dets.at<float>(r, 4);
                if (score < 0.9f) continue;
                faces.emplace_back((int)left, (int)top, (int)width, (int)height);
            }
        }

        if (faces.empty()) {
            fa_msleep(50);
            continue;
        }

        Rect best = faces[0];
        for (const auto &r : faces)
            if (r.area() > best.area()) best = r;

        Mat face = gray(best).clone();
        resize(face, face, Size(200, 200));

        int predicted = -1;
        double confidence = 0.0;
        rec->predict(face, predicted, confidence);

        if (confidence < best_conf)
            best_conf = confidence;

        std::ostringstream oss;
        oss << "Frame " << i << " predicted=" << predicted << " conf=" << confidence;
        log_debug(cfg, oss.str());
    }

    double thr = cfg.lbph_threshold;
    if (method == "eigen") thr = cfg.eigen_threshold;
    else if (method == "fisher") thr = cfg.fisher_threshold;

    bool ok = best_conf <= thr;

    std::ostringstream oss;
    oss << "Auth result: " << (ok ? "SUCCESS" : "FAIL")
        << " conf=" << best_conf << " thr=" << thr
        << " method=" << method;
    log_info(cfg, oss.str());

    if (!ok && cfg.save_failed_images) {
        string img_dir = fa_user_image_dir(cfg, user);
        fa_ensure_dir(img_dir);
        string path = join_path(img_dir, "failed_last.jpg");
        if (!frame.empty()) {
            imwrite(path, frame);
            log_info(cfg, "Saved failed frame to " + path);
        }
    }

    if (!ok) {
        err = "Authentication failed for user " + user;
    }
    return ok;
}

// ------------------------------------------------------------
// CLI tools
// ------------------------------------------------------------

static void print_common_usage() {
    std::cerr << "Options:\n"
              << "  -c, --config <file>    Config file (default "
              << FACIALAUTH_CONFIG_DEFAULT << ")\n"
              << "  -u, --user <user>      Username\n"
              << "  --debug                Force debug=true\n"
              << std::endl;
}

int facial_capture_cli_main(int argc, char **argv) {
    string cfg_path = FACIALAUTH_CONFIG_DEFAULT;
    string user;
    bool debug_override = false;

    for (int i = 1; i < argc; ++i) {
        string a = argv[i];
        if ((a == "-c" || a == "--config") && i + 1 < argc) {
            cfg_path = argv[++i];
        } else if ((a == "-u" || a == "--user") && i + 1 < argc) {
            user = argv[++i];
        } else if (a == "--debug") {
            debug_override = true;
        }
    }

    if (user.empty()) {
        std::cerr << "facial_capture: missing --user\n";
        print_common_usage();
        return 1;
    }

    FacialAuthConfig cfg;
    string err;
    if (!fa_read_config(cfg_path, cfg, err)) {
        std::cerr << "facial_capture: " << err << "\n";
        return 1;
    }
    if (debug_override) cfg.debug = true;

    if (!fa_ensure_dir(cfg.basedir)) {
        std::cerr << "facial_capture: cannot create basedir " << cfg.basedir << "\n";
        return 1;
    }

    string err2;
    if (!fa_capture_images(cfg, user, cfg.capture_count, err2)) {
        std::cerr << "facial_capture: " << err2 << "\n";
        return 1;
    }

    std::cout << "[OK] Capture completed for user " << user << "\n";
    return 0;
}

int facial_training_cli_main(int argc, char **argv) {
    string cfg_path = FACIALAUTH_CONFIG_DEFAULT;
    string user;
    string method = "lbph";
    bool debug_override = false;

    for (int i = 1; i < argc; ++i) {
        string a = argv[i];
        if ((a == "-c" || a == "--config") && i + 1 < argc) {
            cfg_path = argv[++i];
        } else if ((a == "-u" || a == "--user") && i + 1 < argc) {
            user = argv[++i];
        } else if ((a == "-m" || a == "--method") && i + 1 < argc) {
            method = argv[++i];
        } else if (a == "--debug") {
            debug_override = true;
        }
    }

    if (user.empty()) {
        std::cerr << "facial_training: missing --user\n";
        print_common_usage();
        return 1;
    }

    FacialAuthConfig cfg;
    string err;
    if (!fa_read_config(cfg_path, cfg, err)) {
        std::cerr << "facial_training: " << err << "\n";
        return 1;
    }
    if (debug_override) cfg.debug = true;

    if (!fa_ensure_dir(cfg.basedir)) {
        std::cerr << "facial_training: cannot create basedir " << cfg.basedir << "\n";
        return 1;
    }

    string err2;
    if (!fa_train_user(cfg, user, method, err2)) {
        std::cerr << "facial_training: " << err2 << "\n";
        return 1;
    }

    std::cout << "[OK] Model trained: " << fa_user_model_path(cfg, user) << "\n";
    return 0;
}

int facial_test_cli_main(int argc, char **argv) {
    string cfg_path = FACIALAUTH_CONFIG_DEFAULT;
    string user;
    bool debug_override = false;

    for (int i = 1; i < argc; ++i) {
        string a = argv[i];
        if ((a == "-c" || a == "--config") && i + 1 < argc) {
            cfg_path = argv[++i];
        } else if ((a == "-u" || a == "--user") && i + 1 < argc) {
            user = argv[++i];
        } else if (a == "--debug") {
            debug_override = true;
        }
    }

    if (user.empty()) {
        std::cerr << "facial_test: missing --user\n";
        print_common_usage();
        return 1;
    }

    FacialAuthConfig cfg;
    string err;
    if (!fa_read_config(cfg_path, cfg, err)) {
        std::cerr << "facial_test: " << err << "\n";
        return 1;
    }
    if (debug_override) cfg.debug = true;

    double best_conf = 0.0;
    string method;
    string err2;
    bool ok = fa_test_user(cfg, user, best_conf, method, err2);

    if (!ok) {
        std::cerr << "[FAIL] " << err2 << " (conf=" << best_conf << ")\n";
        return 1;
    }

    std::cout << "[OK] Authentication SUCCESS (method=" << method
              << ", conf=" << best_conf << ")\n";
    return 0;
}
