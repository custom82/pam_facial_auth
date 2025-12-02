#include "../include/libfacialauth.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/face.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

#ifdef ENABLE_CUDA
#include <opencv2/core/cuda.hpp>
#endif

#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <dirent.h>

#include <filesystem>
namespace fs = std::filesystem;

// ==========================================================
// Basic helpers
// ==========================================================

static bool file_exists(const std::string &path)
{
    struct stat st;
    return ::stat(path.c_str(), &st) == 0;
}

static bool is_dir(const std::string &path)
{
    struct stat st;
    if (::stat(path.c_str(), &st) != 0) return false;
    return S_ISDIR(st.st_mode);
}

static void ensure_dirs(const std::string &path)
{
    if (path.empty()) return;
    try {
        fs::create_directories(path);
    } catch (...) {}
}

static void sleep_ms_int(int ms)
{
    if (ms <= 0) return;
    usleep((useconds_t)ms * 1000);
}

// ==========================================================
// fa_check_root
// ==========================================================

bool fa_check_root(const std::string &tool_name)
{
    if (::geteuid() != 0) {
        std::cerr << tool_name << " must be run as root.\n";
        return false;
    }
    return true;
}

// ==========================================================
// String utils
// ==========================================================

static inline std::string trim(const std::string &s)
{
    size_t b = 0;
    while (b < s.size() && std::isspace((unsigned char)s[b])) ++b;
    size_t e = s.size();
    while (e > b && std::isspace((unsigned char)s[e - 1])) --e;
    return s.substr(b, e - b);
}

static inline bool starts_with(const std::string &s, const std::string &prefix)
{
    if (s.size() < prefix.size()) return false;
    return std::equal(prefix.begin(), prefix.end(), s.begin());
}

// ==========================================================
// Config loader
// ==========================================================

bool fa_load_config(
    FacialAuthConfig &cfg,
    std::string &logbuf,
    const std::string &path
)
{
    cfg = FacialAuthConfig();
    logbuf.clear();

    std::string cfg_path = path.empty() ? FACIALAUTH_DEFAULT_CONFIG : path;

    std::ifstream f(cfg_path);
    if (!f.is_open()) {
        logbuf += "Cannot open config file: " + cfg_path + "\n";
        return false;
    }

    std::string line;
    int lineno = 0;

    while (std::getline(f, line)) {
        ++lineno;
        std::string s = trim(line);

        if (s.empty() || s[0] == '#')
            continue;

        auto pos = s.find('=');
        if (pos == std::string::npos) {
            logbuf += "Ignoring malformed line " + std::to_string(lineno) + "\n";
            continue;
        }

        std::string key = trim(s.substr(0, pos));
        std::string val = trim(s.substr(pos + 1));

        if (key.empty()) {
            logbuf += "Ignoring empty key at line " + std::to_string(lineno) + "\n";
            continue;
        }

        auto to_bool = [&](const std::string &v, bool &dst) {
            std::string low = v;
            std::transform(low.begin(), low.end(), low.begin(), ::tolower);
            dst = (low == "yes" || low == "true" || low == "1");
        };

        try {
            if (key == "basedir") cfg.basedir = val;
            else if (key == "device") cfg.device = val;
            else if (key == "fallback_device") to_bool(val, cfg.fallback_device);
            else if (key == "width") cfg.width = std::stoi(val);
            else if (key == "height") cfg.height = std::stoi(val);
            else if (key == "frames") cfg.frames = std::stoi(val);
            else if (key == "sleep_ms") cfg.sleep_ms = std::stoi(val);
            else if (key == "debug") to_bool(val, cfg.debug);
            else if (key == "verbose") to_bool(val, cfg.verbose);
            else if (key == "nogui") to_bool(val, cfg.nogui);
            else if (key == "training_method") cfg.training_method = val;
            else if (key == "force_overwrite") to_bool(val, cfg.force_overwrite);
            else if (key == "ignore_failure") to_bool(val, cfg.ignore_failure);
            else if (key == "save_failed_images") to_bool(val, cfg.save_failed_images);
            else if (key == "image_format") cfg.image_format = val;
            else if (key == "detector_profile") cfg.detector_profile = val;
            else if (key == "recognizer_profile") cfg.recognizer_profile = val;
            else if (key == "lbph_threshold") cfg.lbph_threshold = std::stod(val);
            else if (key == "eigen_threshold") cfg.eigen_threshold = std::stod(val);
            else if (key == "fisher_threshold") cfg.fisher_threshold = std::stod(val);
            else if (key == "sface_model") cfg.sface_model = val;
            else if (key == "sface_model_int8") cfg.sface_model_int8 = val;
            else if (key == "haar_model") cfg.haar_cascade_path = val;
            else if (key == "yunet_model") cfg.yunet_model = val;
            else if (key == "yunet_model_int8") cfg.yunet_model_int8 = val;
            else logbuf += "Unknown key '" + key + "' at line " + std::to_string(lineno) + "\n";
        }
        catch (...) {
            logbuf += "Error parsing line " + std::to_string(lineno) + "\n";
        }
    }

    if (cfg.basedir.empty())
        cfg.basedir = "/var/lib/pam_facial_auth";

    return true;
}

// ==========================================================
// Path helpers
// ==========================================================

std::string fa_user_image_dir(const FacialAuthConfig &cfg, const std::string &user)
{
    fs::path base(cfg.basedir.empty() ? "/var/lib/pam_facial_auth" : cfg.basedir);
    fs::path p = base / "images" / user;
    return p.string();
}

std::string fa_user_model_path(const FacialAuthConfig &cfg, const std::string &user)
{
    fs::path base(cfg.basedir.empty() ? "/var/lib/pam_facial_auth" : cfg.basedir);
    fs::path p = base / "models" / (user + ".xml");
    return p.string();
}
// ==========================================================
// Camera helpers
// ==========================================================

static bool open_camera(
    cv::VideoCapture &cap,
    const FacialAuthConfig &cfg,
    std::string &log
)
{
    std::vector<std::string> devs;
    if (!cfg.device.empty())
        devs.push_back(cfg.device);

    if (cfg.fallback_device) {
        for (int i = 0; i < 3; ++i) {
            std::string d = "/dev/video" + std::to_string(i);
            if (std::find(devs.begin(), devs.end(), d) == devs.end())
                devs.push_back(d);
        }
    }

    for (const auto &d : devs) {
        cap.open(d);
        if (cap.isOpened()) {
            if (cfg.debug) log += "Opened camera: " + d + "\n";
            return true;
        } else if (cfg.debug) {
            log += "Failed to open camera: " + d + "\n";
        }
    }

    return false;
}

static bool capture_frame(
    cv::VideoCapture &cap,
    cv::Mat &frame,
    const FacialAuthConfig &cfg,
    std::string &log
)
{
    cap.set(cv::CAP_PROP_FRAME_WIDTH,  cfg.width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.height);

    if (!cap.read(frame) || frame.empty()) {
        log += "Failed to read frame from camera\n";
        return false;
    }

    return true;
}


// ==========================================================
// Model save/load: SFace gallery
// ==========================================================

static bool fa_save_sface_model(
    const FacialAuthConfig &cfg,
    const std::string &profile,
    const std::string &file,
    const std::vector<cv::Mat> &embeds
)
{
    try {
        fs::create_directories(fs::path(file).parent_path());

        cv::FileStorage fs(file, cv::FileStorage::WRITE);
        if (!fs.isOpened()) return false;

        fs << "type" << "sface";
        fs << "version" << 1;

        fs << "recognizer_profile"    << profile;
        fs << "detector_profile"      << cfg.detector_profile;
        fs << "dnn_backend"           << cfg.dnn_backend;
        fs << "dnn_target"            << cfg.dnn_target;
        fs << "sface_fp32_threshold"  << cfg.sface_fp32_threshold;
        fs << "sface_int8_threshold"  << cfg.sface_int8_threshold;
        fs << "width"                 << cfg.width;
        fs << "height"                << cfg.height;
        fs << "frames"                << cfg.frames;

        fs << "embeddings" << "[";
        for (const auto &e : embeds) fs << e;
        fs << "]";

        return true;
    }
    catch (...) { return false; }
}

static bool fa_load_sface_model(
    const std::string &file,
    std::vector<cv::Mat> &embeds
)
{
    embeds.clear();

    try {
        cv::FileStorage fs(file, cv::FileStorage::READ);
        if (!fs.isOpened()) return false;

        std::string type;
        fs["type"] >> type;
        if (type != "sface") return false;

        cv::FileNode emb = fs["embeddings"];
        if (emb.empty() || emb.type() != cv::FileNode::SEQ) return false;

        for (auto it = emb.begin(); it != emb.end(); ++it) {
            cv::Mat m;
            (*it) >> m;
            if (!m.empty()) embeds.push_back(m);
        }

        return !embeds.empty();
    }
    catch (...) { return false; }
}


// ==========================================================
// DNN backend/target mappings
// ==========================================================

static int parse_dnn_backend(const std::string &b)
{
    std::string low = b;
    std::transform(low.begin(), low.end(), low.begin(), ::tolower);

    if (low == "cpu")        return cv::dnn::DNN_BACKEND_OPENCV;
    if (low == "cuda")       return cv::dnn::DNN_BACKEND_CUDA;
    if (low == "cuda_fp16")  return cv::dnn::DNN_BACKEND_CUDA;
    if (low == "opencl")     return cv::dnn::DNN_BACKEND_OPENCV;

    return cv::dnn::DNN_BACKEND_DEFAULT;
}

static int parse_dnn_target(const std::string &t)
{
    std::string low = t;
    std::transform(low.begin(), low.end(), low.begin(), ::tolower);

    if (low == "cpu")        return cv::dnn::DNN_TARGET_CPU;
    if (low == "cuda")       return cv::dnn::DNN_TARGET_CUDA;
    if (low == "cuda_fp16")  return cv::dnn::DNN_TARGET_CUDA_FP16;
    if (low == "opencl")     return cv::dnn::DNN_TARGET_OPENCL;

    return cv::dnn::DNN_TARGET_CPU;
}


// ==========================================================
// SFace ONNX model resolver
// ==========================================================

bool resolve_sface_model(
    const FacialAuthConfig &cfg,
    const std::string &profile,
    std::string &out_model_file,
    std::string &out_resolved_profile
)
{
    std::string p = profile.empty() ? cfg.recognizer_profile : profile;
    if (p.empty()) p = "sface_fp32";

    std::string low = p;
    std::transform(low.begin(), low.end(), low.begin(), ::tolower);

    bool use_int8 = low.find("int8") != std::string::npos;
    std::string key = use_int8 ? "sface_int8" : "sface_fp32";
    std::string path;

    auto it = cfg.recognizer_models.find(key);
    path = (it != cfg.recognizer_models.end()) ? it->second :
    (use_int8 ? cfg.sface_model_int8 : cfg.sface_model);

    if (path.empty()) {
        out_model_file.clear();
        out_resolved_profile.clear();
        return false;
    }

    out_model_file       = path;
    out_resolved_profile = key;
    return true;
}


// ==========================================================
// SFace embedding extraction
// ==========================================================

bool compute_sface_embedding(
    const FacialAuthConfig &cfg,
    const cv::Mat &face,
    const std::string &profile,
    cv::Mat &embedding,
    std::string &log
)
{
    std::string model_path, used_profile;

    if (!resolve_sface_model(cfg, profile, model_path, used_profile)) {
        log += "SFace model not configured.\n";
        return false;
    }

    if (!file_exists(model_path)) {
        log += "Missing SFace ONNX: " + model_path + "\n";
        return false;
    }

    try {
        cv::dnn::Net net = cv::dnn::readNetFromONNX(model_path);
        if (net.empty()) {
            log += "Failed to load SFace model: " + model_path + "\n";
            return false;
        }

        net.setPreferableBackend(parse_dnn_backend(cfg.dnn_backend));
        net.setPreferableTarget(parse_dnn_target(cfg.dnn_target));

        cv::Mat blob = cv::dnn::blobFromImage(
            face,
            1.0 / 255.0,
            cv::Size(112, 112),
                                              cv::Scalar(0,0,0),
                                              true,  // swapRB
                                              false // crop
        );

        net.setInput(blob);
        cv::Mat out = net.forward();
        if (out.empty()) {
            log += "SFace forward(): empty output.\n";
            return false;
        }

        out.reshape(1,1).convertTo(embedding, CV_32F);
        double norm = cv::norm(embedding);
        if (norm > 0.0) embedding /= norm;

        return true;
    }
    catch (const std::exception &e) {
        log += "compute_sface_embedding error: ";
        log += e.what();
        log += "\n";
        return false;
    }
}


// ==========================================================
// DetectorWrapper implementation
// ==========================================================

bool DetectorWrapper::detect(const cv::Mat &frame, cv::Rect &face)
{
    face = cv::Rect();

    if (frame.empty()) return false;

    int W = frame.cols, H = frame.rows;
    if (debug)
        std::cout << "[DEBUG] detect(): " << W << "x" << H << "\n";

    // HAAR
    if (type == DET_HAAR)
    {
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Rect> faces;
        haar.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(30,30));

        if (faces.empty()) return false;
        face = faces[0];

        if (debug) {
            std::cout << "[DEBUG] HAAR face @ "
            << face.x << "," << face.y << " "
            << face.width << "x" << face.height << "\n";
        }

        return true;
    }

    // YuNet DNN
    if (type == DET_YUNET && yunet)
    {
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(640,640));

        try {
            yunet->setInput(cv::dnn::blobFromImage(resized));
            cv::Mat out = yunet->forward();
            if (out.empty()) return false;

            int num = out.size[2];
            float best_score=0.0f;
            cv::Rect best;

            for (int i=0;i<num;i++) {
                float *d = (float*)out.ptr(0,0,i);
                float score = d[14];
                if (score < 0.55f) continue;

                cv::Rect r(
                    int(d[0]*W),
                           int(d[1]*H),
                           int(d[2]*W),
                           int(d[3]*H)
                );

                int minFace = std::min(W,H)/8;
                if (r.width < minFace || r.height < minFace) continue;

                if (score > best_score) {
                    best_score = score;
                    best = r;
                }
            }

            if (best_score <= 0.0f) return false;
            face = best;
            return true;
        }
        catch (...) { return false; }
    }

    return false;
}


// ==========================================================
// Detector initialization
// ==========================================================

static bool init_detector(
    const FacialAuthConfig &cfg,
    DetectorWrapper &det,
    std::string &log
)
{
    std::string profile = cfg.detector_profile;
    if (profile.empty())
        profile = (!cfg.haar_cascade_path.empty() ? "haar" : "yunet_fp32");

    std::string low = profile;
    std::transform(low.begin(), low.end(), low.begin(), ::tolower);

    // HAAR
    if (low == "haar") {
        if (cfg.haar_cascade_path.empty() || !file_exists(cfg.haar_cascade_path)) {
            log += "Missing Haar cascade\n";
            return false;
        }

        if (!det.haar.load(cfg.haar_cascade_path)) {
            log += "Cannot load Haar cascade\n";
            return false;
        }

        det.type = DetectorWrapper::DET_HAAR;
        det.debug = cfg.debug;
        det.model_path = cfg.haar_cascade_path;
        log += "Initialized Haar detector\n";
        return true;
    }

    // YuNet FP32
    if (low == "yunet_fp32") {
        std::string path = cfg.yunet_model;
        if (path.empty() || !file_exists(path)) {
            log += "Missing YuNet FP32 model\n";
            return false;
        }

        try {
            det.yunet = cv::makePtr<cv::dnn::Net>(cv::dnn::readNetFromONNX(path));
            det.type = DetectorWrapper::DET_YUNET;
            det.debug = cfg.debug;
            det.model_path = path;

            det.yunet->setPreferableBackend(parse_dnn_backend(cfg.dnn_backend));
            det.yunet->setPreferableTarget(parse_dnn_target(cfg.dnn_target));

            log += "Initialized YuNet FP32\n";
            return true;
        }
        catch (...) {
            log += "YuNet FP32 init failed\n";
            return false;
        }
    }

    // YuNet INT8
    if (low == "yunet_int8") {
        std::string path = cfg.yunet_model_int8;
        if (path.empty() || !file_exists(path)) {
            log += "Missing YuNet INT8 model\n";
            return false;
        }

        try {
            det.yunet = cv::makePtr<cv::dnn::Net>(cv::dnn::readNetFromONNX(path));
            det.type = DetectorWrapper::DET_YUNET;
            det.debug = cfg.debug;
            det.model_path = path;

            det.yunet->setPreferableBackend(parse_dnn_backend(cfg.dnn_backend));
            det.yunet->setPreferableTarget(parse_dnn_target(cfg.dnn_target));

            log += "Initialized YuNet INT8\n";
            return true;
        }
        catch (...) {
            log += "YuNet INT8 init failed\n";
            return false;
        }
    }

    log += "Unknown detector_profile\n";
    return false;
}
// ==========================================================
// Public API: capture images
// ==========================================================

static bool fa_ensure_directory(const std::string &path, std::string &log)
{
    try {
        if (!fs::exists(path))
            fs::create_directories(path);
        return true;
    }
    catch (const std::exception &e) {
        log += "Cannot create directory '" + path + "': " + e.what() + "\n";
        return false;
    }
}

static int fa_find_next_image_index(const std::string &dir, const std::string &format)
{
    int max_idx = 0;

    if (!fs::exists(dir))
        return 1;

    for (auto &p : fs::directory_iterator(dir)) {
        if (!p.is_regular_file()) continue;

        auto path = p.path();
        if (path.extension() == ("." + format)) {
            try {
                int idx = std::stoi(path.stem().string());
                if (idx > max_idx) max_idx = idx;
            }
            catch (...) {}
        }
    }

    return max_idx + 1;
}

bool fa_capture_images(
    const std::string &user,
    const FacialAuthConfig &cfg,
    const std::string &format,
    std::string &log
)
{
    std::string img_format =
    format.empty()
        ? (cfg.image_format.empty() ? "jpg" : cfg.image_format)
        : format;

    std::string imgdir = fa_user_image_dir(cfg, user);
    if (!fa_ensure_directory(imgdir, log)) {
        log += "[ERROR] Cannot ensure image directory: " + imgdir + "\n";
        return false;
    }

    if (!is_dir(imgdir)) {
        log += "[ERROR] Image path is not a directory: " + imgdir + "\n";
        return false;
    }

    int start_index = fa_find_next_image_index(imgdir, img_format);
    if (cfg.debug) {
        std::cout << "[DEBUG] Will save images into: " << imgdir << "\n";
        std::cout << "[DEBUG] Next available index: " << start_index << "\n";
    }

    cv::VideoCapture cap;
    if (!open_camera(cap, cfg, log)) {
        log += "[ERROR] Cannot open camera.\n";
        return false;
    }

    if (cfg.debug) {
        std::cout << "[DEBUG] Camera opened\n";
        std::cout << "[DEBUG] Setting resolution "
        << cfg.width << "x" << cfg.height << "\n";
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH,  cfg.width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.height);

    DetectorWrapper detector;
    if (!init_detector(cfg, detector, log)) {
        log += "[ERROR] Cannot initialize detector (profile=" +
        cfg.detector_profile + ").\n";
        return false;
    }
    if (cfg.debug) {
        std::cout << "[DEBUG] Detector initialized: "
        << (cfg.detector_profile.empty() ? "auto" : cfg.detector_profile)
        << "\n";
    }

    int saved = 0;
    for (int i = 0; i < cfg.frames; ++i) {
        cv::Mat frame;
        if (!capture_frame(cap, frame, cfg, log)) {
            log += "[ERROR] Invalid frame from camera.\n";
            break;
        }

        if (cfg.verbose) {
            std::cout << "[VERBOSE] Frame " << (i + 1) << "/"
            << cfg.frames << " captured\n";
        }

        cv::Rect face;
        if (!detector.detect(frame, face)) {
            if (cfg.verbose) {
                std::cout << "[VERBOSE] No face detected → frame discarded\n";
            }
            continue;
        }

        if (face.width <= 0 || face.height <= 0) {
            if (cfg.verbose) {
                std::cout << "[VERBOSE] Invalid bounding box → frame discarded\n";
            }
            continue;
        }

        if (cfg.debug) {
            std::cout << "[DEBUG] Face detected: x=" << face.x
            << " y=" << face.y
            << " w=" << face.width
            << " h=" << face.height << "\n";
        }

        int idx = start_index + saved;
        std::string outfile = imgdir + "/" +
        std::to_string(idx) + "." + img_format;

        if (!cv::imwrite(outfile, frame)) {
            log += "[ERROR] Cannot save image: " + outfile + "\n";
        } else {
            ++saved;
            if (cfg.verbose) {
                std::cout << "[VERBOSE] Saved: " << outfile << "\n";
            }
        }

        if (cfg.sleep_ms > 0)
            sleep_ms_int(cfg.sleep_ms);
    }

    if (saved == 0) {
        log += "[WARN] No images saved: no face detected in captured frames.\n";
        return false;
    }

    log += "[INFO] Capture completed. Images saved: " +
    std::to_string(saved) + "\n";
    return true;
}


// ==========================================================
// Training helpers (classic LBPH/Eigen/Fisher)
// ==========================================================

static bool train_classic(
    const std::string &user,
    const FacialAuthConfig &cfg,
    const std::string &method,
    const std::string &imgdir,
    const std::string &model_path,
    bool force_overwrite,
    std::string &log
)
{
    if (!is_dir(imgdir)) {
        log += "Image directory does not exist: " + imgdir + "\n";
        return false;
    }

    std::vector<cv::String> files_jpg;
    std::vector<cv::String> files_png;
    cv::glob(imgdir + "/*.jpg", files_jpg, false);
    cv::glob(imgdir + "/*.png", files_png, false);

    std::vector<cv::String> files = files_jpg;
    files.insert(files.end(), files_png.begin(), files_png.end());

    if (files.empty()) {
        log += "No images found in: " + imgdir + "\n";
        return false;
    }

    std::vector<cv::Mat> faces;
    std::vector<int> labels;

    // Force Haar detection for classic model training
    DetectorWrapper det;
    FacialAuthConfig tmp = cfg;
    tmp.detector_profile = "haar";

    if (!init_detector(tmp, det, log)) {
        log += "Failed to initialize Haar detector for classic training.\n";
        return false;
    }

    for (const auto &fn : files) {
        cv::Mat img = cv::imread(fn);
        if (img.empty()) {
            log += "Cannot read image: " + fn + "\n";
            continue;
        }

        cv::Rect face_rect;
        if (!det.detect(img, face_rect)) {
            log += "No face detected in image: " + fn + "\n";
            continue;
        }

        cv::Mat face = img(face_rect).clone();
        cv::Mat gray;
        cv::cvtColor(face, gray, cv::COLOR_BGR2GRAY);
        cv::resize(gray, gray, cv::Size(92, 112));

        faces.push_back(gray);
        labels.push_back(0);
    }

    if (faces.empty()) {
        log += "No valid faces found for classic training.\n";
        return false;
    }

    // ===============================
    //  Create the classic recognizer
    // ===============================
    cv::Ptr<cv::face::FaceRecognizer> rec;

    std::string mlow = method;
    std::transform(mlow.begin(), mlow.end(), mlow.begin(),
                   [](unsigned char c){ return std::tolower(c); });

    if (mlow == "lbph") {
        rec = cv::face::LBPHFaceRecognizer::create(
            1, 8, 8, 8, cfg.lbph_threshold
        );
    }
    else if (mlow == "eigen" || mlow == "eigenfaces") {
        rec = cv::face::EigenFaceRecognizer::create(
            cfg.eigen_components, cfg.eigen_threshold
        );
    }
    else if (mlow == "fisher" || mlow == "fisherfaces") {
        rec = cv::face::FisherFaceRecognizer::create(
            cfg.fisher_components, cfg.fisher_threshold
        );
    }
    else {
        rec = cv::face::LBPHFaceRecognizer::create(
            1, 8, 8, 8, cfg.lbph_threshold
        );
    }

    if (!rec) {
        log += "Failed to create classic recognizer.\n";
        return false;
    }

    // ===============================
    // Train model
    // ===============================
    try {
        rec->train(faces, labels);
    }
    catch (const std::exception &e) {
        log += std::string("Training failed (classic): ") + e.what() + "\n";
        return false;
    }

    if (file_exists(model_path) && !force_overwrite) {
        log += "Model file already exists (use --force to overwrite): " +
        model_path + "\n";
        return false;
    }

    try {
        rec->save(model_path);
    }
    catch (const std::exception &e) {
        log += std::string("Failed to save model: ") + e.what() + "\n";
        return false;
    }

    log += "Classic model saved to: " + model_path + "\n";
    return true;
}
double dot = a.dot(b);
    double na  = cv::norm(a);
    double nb  = cv::norm(b);
    if (na <= 0.0 || nb <= 0.0) return 0.0;
    return dot / (na * nb);
}

bool fa_test_user(
    const std::string &user,
    const FacialAuthConfig &cfg,
    const std::string &modelPath,
    double &best_conf,
    int &best_label,
    std::string &log,
    double threshold_override
)
{
    best_conf  = 0.0;
    best_label = -1;

    if (!file_exists(modelPath)) {
        log += "Model file not found: " + modelPath + "\n";
        return false;
    }

    std::string method = cfg.training_method;
    std::string rp     = cfg.recognizer_profile;

    std::string method_low = method;
    std::transform(method_low.begin(), method_low.end(), method_low.begin(),
                   [](unsigned char c){ return std::tolower(c); });

    if (method_low == "auto") {
        std::string rp_low = rp;
        std::transform(rp_low.begin(), rp_low.end(), rp_low.begin(),
                       [](unsigned char c){ return std::tolower(c); });
        if (rp_low.rfind("sface", 0) == 0)
            method = "sface";
        else
            method = "lbph";
    }

    std::string mlow = method;
    std::transform(mlow.begin(), mlow.end(), mlow.begin(),
                   [](unsigned char c){ return std::tolower(c); });

    cv::VideoCapture cap;
    if (!open_camera(cap, cfg, log)) {
        log += "fa_test_user: cannot open camera.\n";
        return false;
    }

    DetectorWrapper det;
    if (!init_detector(cfg, det, log)) {
        log += "fa_test_user: cannot initialize detector.\n";
        return false;
    }

    // ------------ SFace branch ------------
    if (mlow == "sface") {
        std::vector<cv::Mat> gallery;
        if (!fa_load_sface_model(modelPath, gallery)) {
            log += "Failed to load SFace model: " + modelPath + "\n";
            return false;
        }

        if (gallery.empty()) {
            log += "SFace model contains zero embeddings.\n";
            return false;
        }

        cv::Mat frame;
        if (!capture_frame(cap, frame, cfg, log)) {
            log += "fa_test_user: failed to capture frame.\n";
            return false;
        }

        cv::Rect face_rect;
        if (!det.detect(frame, face_rect)) {
            log += "No face detected in test frame.\n";
            return false;
        }

        cv::Mat face = frame(face_rect).clone();
        cv::Mat resized;
        cv::resize(face, resized, cv::Size(112, 112));

        cv::Mat emb;
        std::string log_emb;
        if (!compute_sface_embedding(cfg, resized, rp, emb, log_emb)) {
            log += "Failed to compute test embedding.\n";
            log += log_emb;
            return false;
        }

        double best_sim = -1.0;
        int best_idx = -1;

        for (size_t i = 0; i < gallery.size(); ++i) {
            double sim = cosine_similarity(emb, gallery[i]);
            if (sim > best_sim) {
                best_sim = sim;
                best_idx = (int)i;
            }
        }

        best_conf  = best_sim;
        best_label = (best_idx >= 0) ? 0 : -1;

        double thr;
        if (threshold_override >= 0.0) {
            thr = threshold_override;
        } else {
            std::string rp_low = rp;
            std::transform(rp_low.begin(), rp_low.end(), rp_low.begin(),
                           [](unsigned char c){ return std::tolower(c); });
            thr = (rp_low.find("int8") != std::string::npos)
            ? cfg.sface_int8_threshold
            : cfg.sface_fp32_threshold;
        }

        if (best_sim >= thr) {
            log += "SFace similarity " + std::to_string(best_sim) +
            " >= threshold " + std::to_string(thr) + " (accepted)\n";
            return true;
        } else {
            log += "SFace similarity " + std::to_string(best_sim) +
            " < threshold " + std::to_string(thr) + " (rejected)\n";
            return false;
        }
    }

    // ------------ Classic recognizer branch ------------
    cv::Ptr<cv::face::FaceRecognizer> rec;
    try {
        std::string low = cfg.training_method;
        std::transform(low.begin(), low.end(), low.begin(),
                       [](unsigned char c){ return std::tolower(c); });

        if (low == "lbph" || low == "auto") {
            rec = cv::face::LBPHFaceRecognizer::create(
                1, 8, 8, 8, cfg.lbph_threshold
            );
        }
        else if (low == "eigen" || low == "eigenfaces") {
            rec = cv::face::EigenFaceRecognizer::create(
                cfg.eigen_components, cfg.eigen_threshold
            );
        }
        else if (low == "fisher" || low == "fisherfaces") {
            rec = cv::face::FisherFaceRecognizer::create(
                cfg.fisher_components, cfg.fisher_threshold
            );
        }
        else {
            // fallback LBPH
            rec = cv::face::LBPHFaceRecognizer::create(
                1, 8, 8, 8, cfg.lbph_threshold
            );
        }

        rec->read(modelPath);
    }
    catch (const cv::Exception &e) {
        log += "Failed to load classic recognizer model: ";
        log += e.what();
        log += "\n";
        return false;
    }

    cv::Mat frame;
    if (!capture_frame(cap, frame, cfg, log)) {
        log += "fa_test_user: cannot capture frame.\n";
        return false;
    }

    cv::Rect face_rect;
    if (!det.detect(frame, face_rect)) {
        log += "No face detected in test frame.\n";
        return false;
    }

    cv::Mat face = frame(face_rect).clone();
    cv::Mat gray;
    cv::cvtColor(face, gray, cv::COLOR_BGR2GRAY);
    cv::resize(gray, gray, cv::Size(92, 112));

    int label = -1;
    double conf = 0.0;

    try {
        rec->predict(gray, label, conf);
    }
    catch (const std::exception &e) {
        log += "Classic recognizer predict failed: ";
        log += e.what();
        log += "\n";
        return false;
    }

    best_label = label;
    best_conf  = conf;

    log += "Classic recognizer predicted label=" +
    std::to_string(label) +
    " confidence=" + std::to_string(conf) + "\n";

    return true;
}
