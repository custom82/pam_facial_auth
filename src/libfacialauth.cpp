#include "../include/libfacialauth.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/dnn.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <cstdarg>
#include <cctype>
#include <cstring>
#include <cfloat>
#include <sys/stat.h>
#include <unistd.h>

namespace fs = std::filesystem;

using std::string;
using std::vector;

// piccolo helper per compatibilità pre-C++20
static bool str_ends_with(const std::string &s, const std::string &suffix)
{
    if (s.size() < suffix.size())
        return false;
    return std::equal(suffix.rbegin(), suffix.rend(), s.rbegin());
}

// ==========================================================
// Utility varie
// ==========================================================

static string trim(const string &s)
{
    size_t b = 0, e = s.size();
    while (b < e && std::isspace((unsigned char)s[b])) b++;
    while (e > b && std::isspace((unsigned char)s[e - 1])) e--;
    return s.substr(b, e - b);
}

static bool str_to_bool(const string &s, bool defval)
{
    if (s.empty()) return defval;
    string v;
    for (char c : s) v.push_back(std::tolower((unsigned char)c));
    if (v == "1" || v == "true" || v == "yes" || v == "on")
        return true;
    if (v == "0" || v == "false" || v == "no" || v == "off")
        return false;
    return defval;
}

static bool file_exists(const string &path)
{
    struct stat st {};
    return (::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode));
}

static void ensure_dirs(const string &path)
{
    if (path.empty()) return;
    try {
        fs::create_directories(path);
    } catch (...) {
    }
}

static void sleep_ms_int(int ms)
{
    if (ms <= 0) return;
    usleep((useconds_t)ms * 1000);
}

// ==========================================================
// Logging
// ==========================================================

static void log_tool(const FacialAuthConfig &cfg,
                     const char *level,
                     const char *fmt, ...)
{
    char buf[1024];

    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);

    string msg = "[";
    msg += (level ? level : "");
    msg += "] ";
    msg += buf;

    std::cerr << msg << std::endl;

    if (!cfg.log_file.empty()) {
        try {
            ensure_dirs(fs::path(cfg.log_file).parent_path().string());
            std::ofstream out(cfg.log_file, std::ios::app);
            if (out)
                out << msg << std::endl;
        } catch (...) {
        }
    }
}

static inline void log_info (const FacialAuthConfig &cfg, const char *fmt, ...)
{
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    log_tool(cfg, "INFO", "%s", buf);
}

static inline void log_error(const FacialAuthConfig &cfg, const char *fmt, ...)
{
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    log_tool(cfg, "ERROR", "%s", buf);
}

static inline void log_debug(const FacialAuthConfig &cfg, const char *fmt, ...)
{
    if (!cfg.debug) return;
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    log_tool(cfg, "DEBUG", "%s", buf);
}

// ==========================================================
// Path helpers
// ==========================================================

std::string fa_user_image_dir(const FacialAuthConfig &cfg,
                              const std::string &user)
{
    fs::path base = cfg.basedir.empty()
    ? fs::path("/etc/pam_facial_auth")
    : fs::path(cfg.basedir);
    fs::path dir  = base / "images" / user;
    return dir.string();
}

std::string fa_user_model_path(const FacialAuthConfig &cfg,
                               const std::string &user)
{
    fs::path base = cfg.basedir.empty()
    ? fs::path("/etc/pam_facial_auth")
    : fs::path(cfg.basedir);
    fs::path dir  = base / "models";
    fs::path file = dir / (user + ".xml");
    return file.string();
}

// ==========================================================
// Config loader
// ==========================================================

static void apply_dnn_alias(FacialAuthConfig &cfg)
{
    if (!cfg.dnn_backend.empty() && cfg.yunet_backend.empty())
        cfg.yunet_backend = cfg.dnn_backend;
}

bool fa_load_config(FacialAuthConfig &cfg,
                    std::string &logbuf,
                    const std::string &path)
{
    string cfg_path = path.empty() ? string(FACIALAUTH_CONFIG_DEFAULT) : path;

    std::ifstream in(cfg_path);
    if (!in) {
        logbuf += "Cannot open config file: " + cfg_path + "\n";
        return false;
    }

    string line;
    int lineno = 0;
    while (std::getline(in, line)) {
        lineno++;
        string orig = line;
        line = trim(line);
        if (line.empty()) continue;
        if (line[0] == '#') continue;

        string key, val;
        size_t eq = line.find('=');
        if (eq == string::npos) {
            logbuf += "Invalid line " + std::to_string(lineno) + " in config\n";
            continue;
        }

        key = trim(line.substr(0, eq));
        val = trim(line.substr(eq + 1));

        try {
            if (key == "basedir")               cfg.basedir            = val;
            else if (key == "device")           cfg.device             = val;
            else if (key == "fallback_device")  cfg.fallback_device    = str_to_bool(val, cfg.fallback_device);

            else if (key == "width")            cfg.width              = std::max(64, std::stoi(val));
            else if (key == "height")           cfg.height             = std::max(64, std::stoi(val));
            else if (key == "frames")           cfg.frames             = std::max(1,  std::stoi(val));
            else if (key == "sleep_ms")         cfg.sleep_ms           = std::max(0,  std::stoi(val));

            else if (key == "debug")            cfg.debug              = str_to_bool(val, cfg.debug);
            else if (key == "nogui")            cfg.nogui              = str_to_bool(val, cfg.nogui);

            else if (key == "model_path")       cfg.model_path         = val;
            else if (key == "haar_cascade_path" ||
                key == "haar_model")       cfg.haar_cascade_path  = val;
            else if (key == "training_method")  cfg.training_method    = val;
            else if (key == "log_file")         cfg.log_file           = val;

            else if (key == "force_overwrite")  cfg.force_overwrite    = str_to_bool(val, cfg.force_overwrite);
            else if (key == "ignore_failure")   cfg.ignore_failure     = str_to_bool(val, cfg.ignore_failure);

            else if (key == "lbph_threshold")   cfg.lbph_threshold     = std::stod(val);
            else if (key == "eigen_threshold")  cfg.eigen_threshold    = std::stod(val);
            else if (key == "fisher_threshold") cfg.fisher_threshold   = std::stod(val);

            else if (key == "eigen_components") cfg.eigen_components   = std::stoi(val);
            else if (key == "fisher_components")cfg.fisher_components  = std::stoi(val);

            else if (key == "detector_profile") cfg.detector_profile   = val;
            else if (key == "yunet_backend")    cfg.yunet_backend      = val;
            else if (key == "dnn_backend")      cfg.dnn_backend        = val;
            else if (key == "yunet_model")      cfg.yunet_model        = val;
            else if (key == "yunet_model_int8") cfg.yunet_model_int8   = val;

            else if (key == "recognizer_profile") cfg.recognizer_profile = val;
            else if (key == "sface_model")        cfg.sface_model        = val;
            else if (key == "sface_model_int8")   cfg.sface_model_int8   = val;
            else if (key == "sface_threshold")    cfg.sface_threshold    = std::stod(val);
            else if (key == "save_failed_images") cfg.save_failed_images = str_to_bool(val, cfg.save_failed_images);

            else {
                logbuf += "Unknown key at line " + std::to_string(lineno)
                + ": " + key + "\n";
            }
        } catch (const std::exception &e) {
            logbuf += "Error parsing line " + std::to_string(lineno)
            + ": " + orig + " (" + e.what() + ")\n";
        }
    }

    apply_dnn_alias(cfg);
    return true;
}

// ==========================================================
// FaceRecWrapper: LBPH / Eigen / Fisher
// ==========================================================

class FaceRecWrapper {
public:
    explicit FaceRecWrapper(const std::string &modelType = "lbph")
    : modelType_(modelType)
    {}

    bool CreateRecognizer()
    {
        try {
            std::string mt = modelType_;
            for (char &c : mt) c = (char)std::tolower((unsigned char)c);

            if (mt == "eigen") {
                recognizer_ = cv::face::EigenFaceRecognizer::create();
                modelType_  = "eigen";
            } else if (mt == "fisher") {
                recognizer_ = cv::face::FisherFaceRecognizer::create();
                modelType_  = "fisher";
            } else {
                recognizer_ = cv::face::LBPHFaceRecognizer::create();
                modelType_  = "lbph";
            }
            return !recognizer_.empty();
        } catch (...) {
            recognizer_.release();
            return false;
        }
    }

    bool Load(const std::string &file)
    {
        try {
            if (recognizer_.empty() && !CreateRecognizer())
                return false;

            recognizer_->read(file);
            return true;
        } catch (...) {
            return false;
        }
    }

    bool Save(const std::string &file) const
    {
        try {
            ensure_dirs(fs::path(file).parent_path().string());
            recognizer_->write(file);
            return true;
        } catch (...) {
            return false;
        }
    }

    bool Train(const std::vector<cv::Mat> &images,
               const std::vector<int>    &labels)
    {
        if (images.empty() || labels.empty() || images.size() != labels.size())
            return false;

        try {
            if (recognizer_.empty() && !const_cast<FaceRecWrapper*>(this)->CreateRecognizer())
                return false;
            recognizer_->train(images, labels);
            return true;
        } catch (...) {
            return false;
        }
    }

    bool Predict(const cv::Mat &face,
                 int &pred,
                 double &conf) const
                 {
                     if (face.empty()) return false;
                     try {
                         recognizer_->predict(face, pred, conf);
                         return true;
                     } catch (...) {
                         return false;
                     }
                 }

                 bool InitCascade(const std::string &cascadePath)
                 {
                     if (cascadePath.empty())
                         return false;
                     if (!file_exists(cascadePath))
                         return false;

                     try {
                         return faceCascade_.load(cascadePath);
                     } catch (...) {
                         return false;
                     }
                 }

                 bool DetectFace(const cv::Mat &frame, cv::Rect &faceROI)
                 {
                     if (frame.empty()) return false;
                     if (faceCascade_.empty()) return false;

                     cv::Mat gray;
                     cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
                     cv::equalizeHist(gray, gray);

                     std::vector<cv::Rect> faces;
                     faceCascade_.detectMultiScale(
                         gray, faces,
                         1.08, 3,
                         0, cv::Size(60, 60)
                     );

                     if (faces.empty())
                         return false;

                     faceROI = faces[0];
                     return true;
                 }

private:
    std::string modelType_;
    cv::Ptr<cv::face::FaceRecognizer> recognizer_;
    cv::CascadeClassifier             faceCascade_;
};

// ==========================================================
// Detector wrapper: HAAR + YuNet
// ==========================================================

struct DetectorWrapper {
    enum Kind {
        NONE,
        HAAR,
        YUNET
    } kind = NONE;

    cv::CascadeClassifier haar;
    cv::Ptr<cv::FaceDetectorYN> yunet;
};

// ==========================================================
// SFace helpers (via DNN ONNX)
// ==========================================================

static bool fa_save_sface_model(const std::string &file,
                                const std::vector<cv::Mat> &embeds)
{
    try {
        ensure_dirs(fs::path(file).parent_path().string());
        cv::FileStorage fs(file, cv::FileStorage::WRITE);
        if (!fs.isOpened()) return false;

        fs << "type" << "sface";
        fs << "version" << 1;

        fs << "embeddings" << "[";
        for (const auto &e : embeds)
            fs << e;
        fs << "]";

        fs.release();
        return true;
    } catch (...) {
        return false;
    }
}

static bool fa_load_sface_embeddings(const std::string &file,
                                     std::vector<cv::Mat> &embeds)
{
    embeds.clear();

    try {
        cv::FileStorage fs(file, cv::FileStorage::READ);
        if (!fs.isOpened()) return false;

        std::string type;
        fs["type"] >> type;
        if (type != "sface") {
            fs.release();
            return false;
        }

        cv::FileNode embNode = fs["embeddings"];
        for (auto it = embNode.begin(); it != embNode.end(); ++it) {
            cv::Mat v;
            (*it) >> v;
            if (!v.empty())
                embeds.push_back(v.clone());
        }

        fs.release();
        return !embeds.empty();

    } catch (...) {
        return false;
    }
}

static bool load_sface_model_dnn(const FacialAuthConfig &cfg,
                                 const std::string &profile,
                                 cv::dnn::Net &sface_net,
                                 std::string &err)
{
    std::string model_path;

    std::string prof = profile;
    for (char &c : prof) c = (char)std::tolower((unsigned char)c);

    if (prof == "sface_int8") {
        if (!cfg.sface_model_int8.empty() && file_exists(cfg.sface_model_int8))
            model_path = cfg.sface_model_int8;
        else if (!cfg.sface_model.empty() && file_exists(cfg.sface_model))
            model_path = cfg.sface_model;
    } else {
        if (!cfg.sface_model.empty() && file_exists(cfg.sface_model))
            model_path = cfg.sface_model;
        else if (!cfg.sface_model_int8.empty() && file_exists(cfg.sface_model_int8))
            model_path = cfg.sface_model_int8;
    }

    if (model_path.empty()) {
        err = "No SFace model found (check sface_model / sface_model_int8)";
        return false;
    }

    try {
        sface_net = cv::dnn::readNetFromONNX(model_path);
    } catch (const std::exception &e) {
        err = std::string("Failed to load SFace ONNX model: ") + e.what();
        return false;
    } catch (...) {
        err = "Failed to load SFace ONNX model (unknown error)";
        return false;
    }

    if (sface_net.empty()) {
        err = "SFace DNN net is empty after creation";
        return false;
    }

    log_debug(cfg, "Loaded SFace model from '%s'", model_path.c_str());
    return true;
}

static bool sface_feature_from_roi(cv::dnn::Net &net,
                                   const cv::Mat &frame,
                                   const cv::Rect &roi,
                                   cv::Mat &feature)
{
    if (net.empty())
        return false;

    if (frame.empty() || roi.width <= 0 || roi.height <= 0)
        return false;

    cv::Rect r = roi & cv::Rect(0, 0, frame.cols, frame.rows);
    if (r.width <= 0 || r.height <= 0)
        return false;

    cv::Mat face = frame(r).clone();
    if (face.empty())
        return false;

    cv::Mat resized;
    cv::resize(face, resized, cv::Size(112, 112));

    try {
        cv::Mat blob = cv::dnn::blobFromImage(
            resized, 1.0 / 255.0, cv::Size(112, 112),
                                              cv::Scalar(0, 0, 0), true, false
        );

        net.setInput(blob);
        cv::Mat out = net.forward();
        out = out.reshape(1, 1);
        feature = out.clone();
        if (feature.empty()) return false;

        cv::normalize(feature, feature);
        return true;
    } catch (...) {
        return false;
    }
}

// ==========================================================
// Detector init (Haar / YuNet) + debug
// ==========================================================

static bool init_detector(const FacialAuthConfig &cfg,
                          DetectorWrapper &det)
{
    det.kind = DetectorWrapper::NONE;

    std::string detector = cfg.detector_profile;
    for (char &c : detector) c = (char)std::tolower((unsigned char)c);

    log_debug(cfg, "Detector requested profile: '%s'",
              detector.empty() ? "auto" : detector.c_str());

    if (detector.empty() || detector == "auto") {
        if (!cfg.yunet_model.empty() && file_exists(cfg.yunet_model)) {
            detector = "yunet";
            log_debug(cfg, "Detector auto → YUNet (FP32)");
        } else if (!cfg.yunet_model_int8.empty() && file_exists(cfg.yunet_model_int8)) {
            detector = "yunet_int8";
            log_debug(cfg, "Detector auto → YUNet (INT8)");
        } else {
            detector = "haar";
            log_debug(cfg, "Detector auto → Haar Cascade");
        }
    }

    if (detector == "yunet" || detector == "yunet_int8") {
        std::string model_path;

        bool use_int8 = (detector == "yunet_int8"
        || cfg.yunet_backend == "cpu_int8");

        if (use_int8 && !cfg.yunet_model_int8.empty()
            && file_exists(cfg.yunet_model_int8)) {

            model_path = cfg.yunet_model_int8;
        log_debug(cfg,
                  "Detector using YUNet INT8 model: '%s'",
                  model_path.c_str());
            } else if (!cfg.yunet_model.empty() && file_exists(cfg.yunet_model)) {
                model_path = cfg.yunet_model;
                log_debug(cfg,
                          "Detector using YUNet FP32 model: '%s'",
                          model_path.c_str());
            }

            if (!model_path.empty()) {
                try {
                    det.yunet = cv::FaceDetectorYN::create(
                        model_path, "",
                        cv::Size(cfg.width, cfg.height),
                                                           0.9f,
                                                           0.3f,
                                                           5000
                    );
                } catch (...) {
                    det.yunet.release();
                }

                if (!det.yunet.empty()) {
                    det.kind = DetectorWrapper::YUNET;
                    log_debug(cfg, "Detector selected: YUNet");
                    return true;
                }
            }

            log_debug(cfg, "Failed to init YUNet, trying Haar (fallback)");
    }

    if (!cfg.haar_cascade_path.empty() && file_exists(cfg.haar_cascade_path)) {
        try {
            if (det.haar.load(cfg.haar_cascade_path)) {
                det.kind = DetectorWrapper::HAAR;
                log_debug(cfg,
                          "Detector selected: Haar Cascade ('%s')",
                          cfg.haar_cascade_path.c_str());
                return true;
            }
        } catch (...) {
        }
    }

    det.kind = DetectorWrapper::NONE;
    log_error(cfg, "No detector available (YuNet/Haar)");
    return false;
}

// ==========================================================
// Camera helper
// ==========================================================

static bool open_camera(const FacialAuthConfig &cfg,
                        cv::VideoCapture &cap,
                        std::string &dev_used)
{
    std::vector<std::string> devs;
    devs.push_back(cfg.device);
    if (cfg.fallback_device) {
        devs.push_back("/dev/video0");
        devs.push_back("/dev/video1");
        devs.push_back("/dev/video2");
    }

    for (const auto &d : devs) {
        if (d.empty()) continue;
        cap.open(d);
        if (cap.isOpened()) {
            dev_used = d;
            cap.set(cv::CAP_PROP_FRAME_WIDTH,  cfg.width);
            cap.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.height);
            return true;
        }
    }
    return false;
}

// ==========================================================
// Training classico
// ==========================================================

static bool train_classic(const std::string &user,
                          const FacialAuthConfig &cfg,
                          const std::string &img_dir,
                          const std::string &model_path,
                          const std::string &method,
                          bool overwrite,
                          std::string &logbuf)
{
    if (!overwrite && file_exists(model_path)) {
        logbuf += "Model already exists: " + model_path + "\n";
        return false;
    }

    if (cfg.haar_cascade_path.empty() || !file_exists(cfg.haar_cascade_path)) {
        logbuf += "HAAR cascade not configured or missing\n";
        return false;
    }

    std::vector<cv::Mat> images;
    std::vector<int> labels;

    int label = 0;

    if (!fs::exists(img_dir)) {
        logbuf += "Training directory not found: " + img_dir + "\n";
        return false;
    }

    for (auto &entry : fs::directory_iterator(img_dir)) {
        if (!entry.is_regular_file()) continue;
        std::string path = entry.path().string();

        std::string lower = path;
        for (char &c : lower) c = (char)std::tolower((unsigned char)c);
        if (!(str_ends_with(lower, ".jpg")
            || str_ends_with(lower, ".jpeg")
            || str_ends_with(lower, ".png")))
            continue;

        cv::Mat img = cv::imread(path);
        if (img.empty()) continue;

        cv::CascadeClassifier haar;
        if (!haar.load(cfg.haar_cascade_path))
            continue;

        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray, gray);

        std::vector<cv::Rect> faces;
        haar.detectMultiScale(gray, faces, 1.08, 3, 0, cv::Size(60, 60));
        if (faces.empty()) continue;

        cv::Mat roi = gray(faces[0]).clone();
        if (roi.empty()) continue;

        images.push_back(roi);
        labels.push_back(label);
    }

    if (images.empty()) {
        logbuf += "No valid training images for classic model\n";
        return false;
    }

    std::string mt = method;
    for (char &c : mt) c = (char)std::tolower((unsigned char)c);
    if (mt.empty() || mt == "auto")
        mt = "lbph";

    FaceRecWrapper rec(mt);
    if (!rec.CreateRecognizer()) {
        logbuf += "Cannot create classic recognizer\n";
        return false;
    }

    if (!rec.Train(images, labels)) {
        logbuf += "Training failed (classic)\n";
        return false;
    }

    if (!rec.Save(model_path)) {
        logbuf += "Cannot save model to " + model_path + "\n";
        return false;
    }

    return true;
}

// ==========================================================
// fa_capture_images
// ==========================================================

bool fa_capture_images(const std::string &user,
                       const FacialAuthConfig &cfg,
                       std::string &log)
{
    cv::VideoCapture cap(cfg.device, cv::CAP_V4L2);
    if (!cap.isOpened()) {
        log += "[ERROR] Cannot open device: " + cfg.device + "\n";
        return false;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH,  cfg.width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.height);

    std::string userdir = fa_user_image_dir(cfg, user);
    fs::create_directories(userdir);

    DetectorWrapper det;
    if (!init_detector(cfg, det)) {
        log += "[ERROR] Cannot initialize face detector\n";
        return false;
    }

    if (cfg.debug) {
        log += "[DEBUG] Detector active: ";
        log += (det.kind == DetectorWrapper::YUNET ? "YUNet\n" :
        det.kind == DetectorWrapper::HAAR  ? "HAAR\n" :
        "NONE\n");
    }

    int saved = 0;
    int frame_id = 0;

    while (saved < cfg.frames) {

        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            log += "[WARN] Failed to capture frame\n";
            continue;
        }

        std::vector<cv::Rect> faces;

        // ------------------------------
        // Haar detection
        // ------------------------------
        if (det.kind == DetectorWrapper::HAAR) {

            cv::Mat gray;
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            cv::equalizeHist(gray, gray);

            det.haar.detectMultiScale(
                gray,
                faces,
                1.08, 3,
                0,
                cv::Size(60, 60)
            );
        }

        // ------------------------------
        // YuNet detection
        // ------------------------------
        else if (det.kind == DetectorWrapper::YUNET) {

            cv::Mat resized;

            if (frame.size() != cv::Size(cfg.width, cfg.height))
                cv::resize(frame, resized, cv::Size(cfg.width, cfg.height));
            else
                resized = frame;

            cv::Mat dets;
            det.yunet->detect(resized, dets);

            for (int i = 0; i < dets.rows; i++) {
                float score = dets.at<float>(i, 4);
                if (score < 0.6f) continue;

                int x = dets.at<float>(i, 0);
                int y = dets.at<float>(i, 1);
                int w = dets.at<float>(i, 2);
                int h = dets.at<float>(i, 3);

                faces.emplace_back(x, y, w, h);
            }
        }

        if (cfg.debug) {
            log += "[DEBUG] Frame ";
            log += std::to_string(frame_id);
            log += ": detected ";
            log += std::to_string(faces.size());
            log += " faces\n";
        }

        frame_id++;

        if (faces.empty())
            continue;

        // Prendi faccia più grande
        cv::Rect roi = faces[0];
        for (const auto &f : faces)
            if (f.area() > roi.area())
                roi = f;

        // Clampa ROI
        roi = roi & cv::Rect(0, 0, frame.cols, frame.rows);

        if (cfg.debug) {
            log += "[DEBUG] ROI selected: x=" + std::to_string(roi.x)
            + " y=" + std::to_string(roi.y)
            + " w=" + std::to_string(roi.width)
            + " h=" + std::to_string(roi.height) + "\n";
        }

        cv::Mat face = frame(roi).clone();
        if (face.empty()) {
            if (cfg.debug) log += "[DEBUG] Empty ROI after cropping\n";
            continue;
        }

        char name[256];
        snprintf(name, sizeof(name),
                 "%s/img_%04d.jpg", userdir.c_str(), saved);

        if (!cv::imwrite(name, face)) {
            log += "[ERROR] Cannot save image: ";
            log += name;
            log += "\n";
            continue;
        }

        log += "[INFO] Saved face: ";
        log += name;
        log += " (";
        log += std::to_string(face.cols) + "x" + std::to_string(face.rows);
        log += ")\n";

        if (cfg.debug) {
            log += "[DEBUG] Image saved from ROI: ";
            log += std::to_string(roi.x) + "," + std::to_string(roi.y) +
            "," + std::to_string(roi.width) + "," +
            std::to_string(roi.height) + "\n";
        }

        saved++;

        if (cfg.sleep_ms > 0)
            usleep(cfg.sleep_ms * 1000);
    }

    return true;
}


// ==========================================================
// fa_train_user
// ==========================================================

bool fa_train_user(const std::string &user,
                   const FacialAuthConfig &cfg,
                   std::string &logbuf)
{
    std::string img_dir    = fa_user_image_dir(cfg, user);
    std::string model_path = cfg.model_path.empty()
    ? fa_user_model_path(cfg, user)
    : cfg.model_path;

    if (!fs::exists(img_dir)) {
        logbuf += "Image directory not found: " + img_dir + "\n";
        return false;
    }

    std::string rp = cfg.recognizer_profile;
    for (char &c : rp) c = (char)std::tolower((unsigned char)c);
    if (rp.empty())
        rp = "sface";

    bool use_sface = (rp == "sface" || rp == "sface_int8");

    if (!use_sface) {
        return train_classic(user, cfg, img_dir, model_path,
                             cfg.training_method, cfg.force_overwrite, logbuf);
    }

    cv::dnn::Net sface;
    std::string err;
    if (!load_sface_model_dnn(cfg, rp, sface, err)) {
        logbuf += err + "\n";
        return false;
    }

    DetectorWrapper det;
    if (!init_detector(cfg, det)) {
        logbuf += "Cannot init detector (YuNet/HAAR) for SFace training\n";
        return false;
    }

    std::vector<cv::Mat> embeddings;

    for (auto &entry : fs::directory_iterator(img_dir)) {
        if (!entry.is_regular_file()) continue;
        std::string path = entry.path().string();

        std::string lower = path;
        for (char &c : lower) c = (char)std::tolower((unsigned char)c);
        if (!(str_ends_with(lower, ".jpg")
            || str_ends_with(lower, ".jpeg")
            || str_ends_with(lower, ".png")))
            continue;

        cv::Mat img = cv::imread(path);
        if (img.empty()) continue;

        cv::Rect roi;
        bool have_face = false;

        if (det.kind == DetectorWrapper::HAAR) {
            cv::Mat gray;
            cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
            cv::equalizeHist(gray, gray);
            std::vector<cv::Rect> faces;
            det.haar.detectMultiScale(gray, faces, 1.08, 3, 0, cv::Size(60, 60));
            if (!faces.empty()) {
                roi = faces[0];
                have_face = true;
            }
        } else if (det.kind == DetectorWrapper::YUNET) {
            cv::Mat resized;
            if (img.size() != cv::Size(cfg.width, cfg.height))
                cv::resize(img, resized, cv::Size(cfg.width, cfg.height));
            else
                resized = img;

            cv::Mat dets;
            det.yunet->detect(resized, dets);
            int best_i = -1;
            float best_score = 0.0f;
            for (int j = 0; j < dets.rows; ++j) {
                float score = dets.at<float>(j, 4);
                if (score > 0.9f && score > best_score) {
                    best_score = score;
                    best_i = j;
                }
            }
            if (best_i >= 0) {
                float x = dets.at<float>(best_i, 0);
                float y = dets.at<float>(best_i, 1);
                float w = dets.at<float>(best_i, 2);
                float h = dets.at<float>(best_i, 3);
                roi = cv::Rect(cv::Point2f(x, y), cv::Size2f(w, h));
                have_face = true;
            }
        }

        if (!have_face) continue;

        cv::Mat feat;
        if (!sface_feature_from_roi(sface, img, roi, feat))
            continue;

        embeddings.push_back(feat.clone());
    }

    if (embeddings.empty()) {
        logbuf += "No embeddings extracted for SFace model\n";
        return false;
    }

    if (!fa_save_sface_model(model_path, embeddings)) {
        logbuf += "Cannot save SFace model: " + model_path + "\n";
        return false;
    }

    log_info(cfg, "Saved SFace model (%zu embeddings) to %s",
             embeddings.size(), model_path.c_str());
    return true;
}

// ==========================================================
// fa_test_user
// ==========================================================

bool fa_test_user(const std::string &user,
                  const FacialAuthConfig &cfg,
                  const std::string &modelPath,
                  double &best_conf,
                  int &best_label,
                  std::string &logbuf,
                  double threshold_override)
{
    std::string rp = cfg.recognizer_profile;
    for (char &c : rp) c = (char)std::tolower((unsigned char)c);
    if (rp.empty())
        rp = "sface";

    bool use_sface = (rp == "sface" || rp == "sface_int8");

    if (use_sface) {
        cv::dnn::Net sface;
        std::string err;
        if (!load_sface_model_dnn(cfg, rp, sface, err)) {
            logbuf += err + "\n";
            return false;
        }

        DetectorWrapper det;
        if (!init_detector(cfg, det)) {
            logbuf += "Cannot init detector (YuNet/HAAR) for SFace test\n";
            return false;
        }

        std::vector<cv::Mat> gallery_feats;

        std::string model_file = modelPath;
        if (!model_file.empty() && file_exists(model_file)) {
            if (fa_load_sface_embeddings(model_file, gallery_feats)) {
                log_info(cfg, "Loaded %zu SFace embeddings from model",
                         gallery_feats.size());
            }
        }

        if (gallery_feats.empty()) {
            std::string img_dir = fa_user_image_dir(cfg, user);
            if (!fs::exists(img_dir)) {
                logbuf += "No image directory for user: " + img_dir + "\n";
                return false;
            }

            for (auto &entry : fs::directory_iterator(img_dir)) {
                if (!entry.is_regular_file()) continue;
                std::string path = entry.path().string();

                std::string lower = path;
                for (char &c : lower) c = (char)std::tolower((unsigned char)c);
                if (!(str_ends_with(lower, ".jpg")
                    || str_ends_with(lower, ".jpeg")
                    || str_ends_with(lower, ".png")))
                    continue;

                cv::Mat img = cv::imread(path);
                if (img.empty()) continue;

                cv::Rect roi;
                bool have_face = false;

                if (det.kind == DetectorWrapper::HAAR) {
                    cv::Mat gray;
                    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
                    cv::equalizeHist(gray, gray);
                    std::vector<cv::Rect> faces;
                    det.haar.detectMultiScale(
                        gray, faces, 1.08, 3, 0, cv::Size(60, 60)
                    );
                    if (!faces.empty()) {
                        roi = faces[0];
                        have_face = true;
                    }
                } else if (det.kind == DetectorWrapper::YUNET) {
                    cv::Mat resized;
                    if (img.size() != cv::Size(cfg.width, cfg.height))
                        cv::resize(img, resized, cv::Size(cfg.width, cfg.height));
                    else
                        resized = img;

                    cv::Mat dets;
                    det.yunet->detect(resized, dets);
                    int best_i = -1;
                    float best_score = 0.0f;
                    for (int j = 0; j < dets.rows; ++j) {
                        float score = dets.at<float>(j, 4);
                        if (score > 0.9f && score > best_score) {
                            best_score = score;
                            best_i = j;
                        }
                    }
                    if (best_i >= 0) {
                        float x = dets.at<float>(best_i, 0);
                        float y = dets.at<float>(best_i, 1);
                        float w = dets.at<float>(best_i, 2);
                        float h = dets.at<float>(best_i, 3);
                        roi = cv::Rect(cv::Point2f(x, y),
                                       cv::Size2f(w, h));
                        have_face = true;
                    }
                }

                if (!have_face) continue;

                cv::Mat feat;
                if (!sface_feature_from_roi(sface, img, roi, feat))
                    continue;

                gallery_feats.push_back(feat.clone());
            }

            if (gallery_feats.empty()) {
                logbuf += "No SFace gallery features for user\n";
                return false;
            }
        }

        cv::VideoCapture cap;
        std::string dev;
        if (!open_camera(cfg, cap, dev)) {
            logbuf += "Cannot open camera for SFace test\n";
            return false;
        }

        log_info(cfg, "Testing user %s (SFace) on device %s",
                 user.c_str(), dev.c_str());

        best_conf  = -1.0;
        best_label = 0;

        double threshold = cfg.sface_threshold;
        if (threshold_override > 0.0)
            threshold = threshold_override;

        cv::Mat frame;

        for (int i = 0; i < cfg.frames; ++i) {
            cap >> frame;
            if (frame.empty()) {
                sleep_ms_int(cfg.sleep_ms);
                continue;
            }

            cv::Rect roi;
            bool have_face = false;

            if (det.kind == DetectorWrapper::HAAR) {
                cv::Mat gray;
                cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
                cv::equalizeHist(gray, gray);
                std::vector<cv::Rect> faces;
                det.haar.detectMultiScale(
                    gray, faces, 1.08, 3, 0, cv::Size(60, 60)
                );
                if (!faces.empty()) {
                    roi = faces[0];
                    have_face = true;
                }
            } else if (det.kind == DetectorWrapper::YUNET) {
                cv::Mat resized;
                if (frame.size() != cv::Size(cfg.width, cfg.height))
                    cv::resize(frame, resized, cv::Size(cfg.width, cfg.height));
                else
                    resized = frame;

                cv::Mat dets;
                det.yunet->detect(resized, dets);
                int best_i = -1;
                float best_score = 0.0f;
                for (int j = 0; j < dets.rows; ++j) {
                    float score = dets.at<float>(j, 4);
                    if (score > 0.9f && score > best_score) {
                        best_score = score;
                        best_i = j;
                    }
                }
                if (best_i >= 0) {
                    float x = dets.at<float>(best_i, 0);
                    float y = dets.at<float>(best_i, 1);
                    float w = dets.at<float>(best_i, 2);
                    float h = dets.at<float>(best_i, 3);
                    roi = cv::Rect(cv::Point2f(x, y), cv::Size2f(w, h));
                    have_face = true;
                }
            }

            if (!have_face) {
                sleep_ms_int(cfg.sleep_ms);
                continue;
            }

            cv::Mat feat;
            if (!sface_feature_from_roi(sface, frame, roi, feat)) {
                sleep_ms_int(cfg.sleep_ms);
                continue;
            }

            double best_sim_frame = -1.0;
            for (const auto &g : gallery_feats) {
                if (g.empty() || feat.size() != g.size())
                    continue;
                double sim = feat.dot(g);
                if (sim > best_sim_frame)
                    best_sim_frame = sim;
            }

            if (best_sim_frame > best_conf)
                best_conf = best_sim_frame;

            if (best_sim_frame >= threshold) {
                log_info(cfg,
                         "Auth success (SFace): sim=%.3f >= %.3f",
                         best_sim_frame, threshold);
                return true;
            }

            sleep_ms_int(cfg.sleep_ms);
        }

        log_info(cfg,
                 "Auth failed (SFace): best_sim=%.3f threshold=%.3f",
                 best_conf, threshold);
        return false;
    }

    // -------------------------------
    // Modalità classica LBPH/EIGEN/FISHER
    // -------------------------------
    std::string model_file =
    modelPath.empty() ? fa_user_model_path(cfg, user) : modelPath;

    if (!file_exists(model_file)) {
        logbuf += "Model file missing: " + model_file + "\n";
        return false;
    }

    std::string mt = cfg.training_method;
    for (char &c : mt) c = (char)std::tolower((unsigned char)c);
    if (mt.empty() || mt == "auto")
        mt = "lbph";

    FaceRecWrapper rec(mt);

    if (!rec.CreateRecognizer()) {
        logbuf += "Recognizer creation failed (" + mt + ")\n";
        return false;
    }

    if (!rec.Load(model_file)) {
        logbuf += "Cannot load model: " + model_file + "\n";
        return false;
    }

    if (!rec.InitCascade(cfg.haar_cascade_path)) {
        logbuf += "Cannot load HAAR cascade for testing\n";
        return false;
    }

    cv::VideoCapture cap;
    std::string dev;
    if (!open_camera(cfg, cap, dev)) {
        logbuf += "Cannot open camera for classic test\n";
        return false;
    }

    log_info(cfg, "Testing user %s (model=%s) on device %s",
             user.c_str(), mt.c_str(), dev.c_str());

    best_conf  = 1e9;
    best_label = -1;

    double threshold = cfg.lbph_threshold;
    if (mt == "eigen")  threshold = cfg.eigen_threshold;
    if (mt == "fisher") threshold = cfg.fisher_threshold;

    if (threshold_override > 0.0)
        threshold = threshold_override;

    cv::Mat frame;

    for (int i = 0; i < cfg.frames; ++i) {
        cap >> frame;
        if (frame.empty()) {
            sleep_ms_int(cfg.sleep_ms);
            continue;
        }

        cv::Rect roi;
        if (!rec.DetectFace(frame, roi)) {
            sleep_ms_int(cfg.sleep_ms);
            continue;
        }

        cv::Mat face = frame(roi).clone();
        if (face.empty()) {
            sleep_ms_int(cfg.sleep_ms);
            continue;
        }

        cv::Mat gray;
        cv::cvtColor(face, gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray, gray);

        int    label = -1;
        double conf  = 0.0;
        if (!rec.Predict(gray, label, conf)) {
            sleep_ms_int(cfg.sleep_ms);
            continue;
        }

        if (conf < best_conf) {
            best_conf  = conf;
            best_label = label;
        }

        if (conf <= threshold) {
            log_info(cfg,
                     "Auth success (model=%s): conf=%.2f <= %.2f",
                     mt.c_str(), conf, threshold);
            return true;
        }

        sleep_ms_int(cfg.sleep_ms);
    }

    log_info(cfg,
             "Auth failed (model=%s): best_conf=%.2f threshold=%.2f",
             mt.c_str(), best_conf, threshold);

    return false;
}

// ==========================================================
// fa_check_root
// ==========================================================

bool fa_check_root(const char *tool_name)
{
    if (::geteuid() != 0) {
        std::cerr << tool_name << ": must be run as root\n";
        return false;
    }
    return true;
}

// ==========================================================
// CLI WRAPPERS
// ==========================================================

static void print_common_usage(const char *prog)
{
    std::cout << "Usage: " << prog << " -u USER [options]\n";
}

int facial_capture_main (int argc, char *argv[])
{
    const char *prog = "facial_capture";

    if (!fa_check_root(prog))
        return 1;

    std::string user;
    std::string cfg_path;
    std::string format;

    bool clean  = false;
    bool reset  = false;
    bool force  = false;

    std::string opt_device;
    std::string opt_detector;
    bool opt_debug = false;
    bool opt_nogui = false;
    int  opt_width = -1;
    int  opt_height = -1;
    int  opt_frames = -1;
    int  opt_sleep = -1;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];

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
            force = true;
        else if (a == "-g" || a == "--nogui")
            opt_nogui = true;
        else if (a == "-v" || a == "--debug")
            opt_debug = true;
        else if (a == "--detector" && i + 1 < argc)
            opt_detector = argv[++i];
        else if (a == "--clean")
            clean = true;
        else if (a == "--reset")
            reset = true;
        else if (a == "--format" && i + 1 < argc)
            format = argv[++i];
        else if (a == "--help") {
            print_common_usage(prog);
            return 0;
        }
    }

    if (user.empty()) {
        print_common_usage(prog);
        return 1;
    }

    FacialAuthConfig cfg;
    std::string logbuf;

    fa_load_config(cfg, logbuf,
                   cfg_path.empty() ? FACIALAUTH_CONFIG_DEFAULT : cfg_path);

    if (!logbuf.empty())
        std::cerr << logbuf;
    logbuf.clear();

    if (!opt_device.empty())   cfg.device           = opt_device;
    if (!opt_detector.empty()) cfg.detector_profile = opt_detector;
    if (opt_width  > 0)        cfg.width            = opt_width;
    if (opt_height > 0)        cfg.height           = opt_height;
    if (opt_frames > 0)        cfg.frames           = opt_frames;
    if (opt_sleep >= 0)        cfg.sleep_ms         = opt_sleep;
    if (opt_debug)             cfg.debug            = true;
    if (opt_nogui)             cfg.nogui            = true;

    std::string user_img_dir = fa_user_image_dir(cfg, user);
    std::string user_model   = fa_user_model_path(cfg, user);

    if (reset) {
        fs::remove_all(user_img_dir);
        if (fs::exists(user_model))
            fs::remove(user_model);
        return 0;
    }

    if (clean) {
        fs::remove_all(user_img_dir);
        return 0;
    }

    if (force)
        fs::remove_all(user_img_dir);

    bool ok = fa_capture_images(user, cfg, logbuf);

    if (!logbuf.empty())
        std::cerr << logbuf;

    return ok ? 0 : 1;
}

int facial_training_cli_main(int argc, char *argv[])
{
    const char *prog = "facial_training";

    if (!fa_check_root(prog))
        return 1;

    std::string user;
    std::string cfg_path;

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

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];

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
        else if (a == "-v" || a == "--debug")
            opt_debug = true;
        else if (a == "--detector" && i + 1 < argc)
            opt_detector = argv[++i];
        else if (a == "--clean")
            opt_clean = true;
        else if (a == "--reset")
            opt_reset = true;
        else if (a == "--help") {
            print_common_usage(prog);
            return 0;
        }
    }

    if (user.empty()) {
        print_common_usage(prog);
        return 1;
    }

    FacialAuthConfig cfg;
    std::string logbuf;

    fa_load_config(cfg, logbuf,
                   cfg_path.empty() ? FACIALAUTH_CONFIG_DEFAULT : cfg_path);

    if (!logbuf.empty())
        std::cerr << logbuf;
    logbuf.clear();

    if (!opt_device.empty())   cfg.device           = opt_device;
    if (!opt_detector.empty()) cfg.detector_profile = opt_detector;
    if (opt_width  > 0)        cfg.width            = opt_width;
    if (opt_height > 0)        cfg.height           = opt_height;
    if (opt_frames > 0)        cfg.frames           = opt_frames;
    if (opt_sleep >= 0)        cfg.sleep_ms         = opt_sleep;
    if (opt_debug)             cfg.debug            = true;
    if (opt_nogui)             cfg.nogui            = true;
    if (opt_force)             cfg.force_overwrite  = true;

    std::string img_dir  = fa_user_image_dir(cfg, user);
    std::string mdl_path = fa_user_model_path(cfg, user);

    if (opt_reset) {
        fs::remove_all(img_dir);
        if (fs::exists(mdl_path))
            fs::remove(mdl_path);
        return 0;
    }

    if (opt_clean) {
        fs::remove_all(img_dir);
        return 0;
    }

    bool ok = fa_train_user(user, cfg, logbuf);

    if (!logbuf.empty())
        std::cerr << logbuf;

    return ok ? 0 : 1;
}

int facial_test_cli_main(int argc, char *argv[])
{
    const char *prog = "facial_test";

    if (!fa_check_root(prog))
        return 1;

    std::string user;
    std::string cfg_path;

    std::string opt_device;
    std::string opt_detector;
    bool opt_debug = false;
    bool opt_nogui = false;

    int opt_width  = -1;
    int opt_height = -1;
    int opt_frames = -1;
    int opt_sleep  = -1;

    double opt_threshold = -1.0;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];

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
        else if (a == "-g" || a == "--nogui")
            opt_nogui = true;
        else if (a == "-v" || a == "--debug")
            opt_debug = true;
        else if (a == "--detector" && i + 1 < argc)
            opt_detector = argv[++i];
        else if ((a == "-t" || a == "--threshold") && i + 1 < argc)
            opt_threshold = atof(argv[++i]);
        else if (a == "--help") {
            print_common_usage(prog);
            return 0;
        }
    }

    if (user.empty()) {
        print_common_usage(prog);
        return 1;
    }

    FacialAuthConfig cfg;
    std::string logbuf;

    fa_load_config(cfg, logbuf,
                   cfg_path.empty() ? FACIALAUTH_CONFIG_DEFAULT : cfg_path);

    if (!logbuf.empty())
        std::cerr << logbuf;
    logbuf.clear();

    if (!opt_device.empty())   cfg.device           = opt_device;
    if (!opt_detector.empty()) cfg.detector_profile = opt_detector;
    if (opt_debug)             cfg.debug            = true;
    if (opt_nogui)             cfg.nogui            = true;
    if (opt_width  > 0)        cfg.width            = opt_width;
    if (opt_height > 0)        cfg.height           = opt_height;
    if (opt_frames > 0)        cfg.frames           = opt_frames;
    if (opt_sleep >= 0)        cfg.sleep_ms         = opt_sleep;

    double best_conf  = 0.0;
    int    best_label = -1;

    bool ok = fa_test_user(
        user,
        cfg,
        cfg.model_path,
        best_conf,
        best_label,
        logbuf,
        opt_threshold
    );

    if (!logbuf.empty())
        std::cerr << logbuf;

    return ok ? 0 : 1;
}
