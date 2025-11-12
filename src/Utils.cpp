#include "../include/Utils.h"
#include <sys/stat.h>
#include <unistd.h>
#include <cstdarg>
#include <fstream>
#include <sstream>
#include <iostream>

std::string trim(const std::string &s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

bool str_to_bool(const std::string &s, bool defval) {
    auto t = trim(s);
    for (auto &c : t) c = ::tolower(c);
    if (t == "1" || t == "true" || t == "yes" || t == "on") return true;
    if (t == "0" || t == "false" || t == "no"  || t == "off") return false;
    return defval;
}

bool read_kv_config(const std::string &path, FacialAuthConfig &cfg, std::string *logbuf) {
    std::ifstream in(path);
    if (!in.is_open()) {
        if (logbuf) *logbuf += "Config not found: " + path + "\n";
        return false;
    }
    if (logbuf) *logbuf += "Reading config: " + path + "\n";
    std::string line;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        if (logbuf) *logbuf += "LINE: " + line + "\n";
        std::istringstream iss(line);
        std::string key, val;
        if (!(iss >> key)) continue;
        std::getline(iss, val);
        val = trim(val);
        if (key == "device") cfg.device = val;
        else if (key == "width") cfg.width = std::max(64, std::stoi(val));
        else if (key == "height") cfg.height = std::max(64, std::stoi(val));
        else if (key == "threshold") cfg.threshold = std::stod(val);
        else if (key == "timeout") cfg.timeout = std::max(1, std::stoi(val));
        else if (key == "nogui") cfg.nogui = str_to_bool(val, cfg.nogui);
        else if (key == "debug") cfg.debug = str_to_bool(val, cfg.debug);
        else if (key == "model") cfg.model = val;                   // lbph/eigen/fisher
        else if (key == "detector") cfg.detector = val;             // auto/haar/dnn
        else if (key == "model_format") cfg.model_format = val;     // xml/yaml/onnx/both
        else if (key == "frames") cfg.frames = std::max(1, std::stoi(val));
        else if (key == "fallback_device") cfg.fallback_device = str_to_bool(val, cfg.fallback_device);
        else if (key == "model_path") cfg.model_path = val;
    }
    return true;
}

void ensure_dirs(const std::string &path) {
    std::string p = path;
    if (p.empty()) return;
    std::string cur;
    std::stringstream ss(p);
    std::string tok;
    if (p[0] == '/') cur = "/";
    while (std::getline(ss, tok, '/')) {
        if (tok.empty()) continue;
        if (cur.size() > 1) cur += "/";
        cur += tok;
        ::mkdir(cur.c_str(), 0755);
    }
}

bool file_exists(const std::string &path) {
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

std::string join_path(const std::string &a, const std::string &b) {
    if (a.empty()) return b;
    if (b.empty()) return a;
    if (a.back() == '/') return a + b;
    return a + "/" + b;
}

void sleep_ms(int ms) { ::usleep(ms * 1000); }

void log_tool(bool debug, const char* level, const char* fmt, ...) {
    if (!debug && std::string(level) == "DEBUG") return;
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    std::cerr << "[FA-" << level << "] " << buf << std::endl;
}

// Apertura camera con fallback
bool open_camera(const FacialAuthConfig &cfg, cv::VideoCapture &cap, std::string &device_used) {
    device_used = cfg.device;
    cap.open(cfg.device);
    if (!cap.isOpened() && cfg.fallback_device) {
        log_tool(cfg.debug, "WARN", "Primary device %s failed, trying /dev/video1", cfg.device.c_str());
        cap.open("/dev/video1");
        if (cap.isOpened()) device_used = "/dev/video1";
    }
    if (!cap.isOpened()) return false;
    (void)cap.set(cv::CAP_PROP_FRAME_WIDTH, cfg.width);
    (void)cap.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.height);
    return true;
}

// Detector
static bool detect_dnn_impl(const cv::Mat &frame, cv::dnn::Net &net, cv::Rect &roi) {
    if (net.empty()) return false;
    cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0, cv::Size(300,300),
                                          cv::Scalar(104,177,123), false, false);
    net.setInput(blob);
    cv::Mat out = net.forward();
    cv::Mat det(out.size[2], out.size[3], CV_32F, out.ptr<float>());
    float best = 0.f;
    cv::Rect bestR;
    for (int i=0;i<det.rows;i++){
        float conf = det.at<float>(i,2);
        if (conf < 0.5f) continue;
        int x1 = int(det.at<float>(i,3) * frame.cols);
        int y1 = int(det.at<float>(i,4) * frame.rows);
        int x2 = int(det.at<float>(i,5) * frame.cols);
        int y2 = int(det.at<float>(i,6) * frame.rows);
        cv::Rect r(cv::Point(x1,y1), cv::Point(x2,y2));
        r &= cv::Rect(0,0,frame.cols,frame.rows);
        if (r.area() > best) { best = (float)r.area(); bestR = r; }
    }
    if (best > 0) { roi = bestR; return true; }
    return false;
}

static bool detect_haar_impl(const cv::Mat &frame, cv::CascadeClassifier &haar, cv::Rect &roi) {
    if (haar.empty()) return false;
    std::vector<cv::Rect> faces;
    cv::Mat gray; cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(gray, gray);
    haar.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(80,80));
    if (faces.empty()) return false;
    // prendi il più grande
    size_t idx = 0; int best = 0;
    for (size_t i=0;i<faces.size();++i) {
        if (faces[i].area() > best) { best = faces[i].area(); idx = i; }
    }
    roi = faces[idx];
    return true;
}

bool detect_face(const FacialAuthConfig &cfg, const cv::Mat &frame, cv::Rect &face_roi,
                 cv::CascadeClassifier &haar, cv::dnn::Net &dnn)
{
    if (!frame.data) return false;
    if (cfg.detector == "dnn") {
        if (detect_dnn_impl(frame, dnn, face_roi)) return true;
        return false;
    }
    if (cfg.detector == "haar") {
        if (detect_haar_impl(frame, haar, face_roi)) return true;
        return false;
    }
    // auto
    if (detect_dnn_impl(frame, dnn, face_roi)) return true;
    if (detect_haar_impl(frame, haar, face_roi)) return true;
    return false;
}

void load_detectors(const FacialAuthConfig &cfg,
                    cv::CascadeClassifier &haar, cv::dnn::Net &dnn,
                    bool &use_dnn, std::string &log)
{
    log.clear();
    use_dnn = false;

    // Prova DNN se richiesto/auto e se i modelli esistono
    std::string dpath = join_path(cfg.model_path, "dnn");
    std::string proto = join_path(dpath, "deploy.prototxt");
    std::string weight = join_path(dpath, "res10_300x300_ssd_iter_140000_fp16.caffemodel");
    if ((cfg.detector == "dnn" || cfg.detector == "auto") && file_exists(proto) && file_exists(weight)) {
        try {
            dnn = cv::dnn::readNetFromCaffe(proto, weight);
            if (!dnn.empty()) {
                use_dnn = true;
                log += "DNN detector loaded\n";
            }
        } catch (...) {
            log += "DNN load failed, fallback to HAAR\n";
        }
    }
    // Sempre tenta Haar
    std::string haarp = join_path(cfg.model_path, "haarcascades/haarcascade_frontalface_default.xml");
    if (!file_exists(haarp)) {
        // fallback: path built-in se disponibile (alcune distro)
        haarp = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml";
    }
    if (file_exists(haarp)) {
        if (haar.load(haarp)) log += "HAAR detector loaded: " + haarp + "\n";
        else log += "HAAR detector failed to load\n";
    } else {
        log += "HAAR cascade not found\n";
    }
}
