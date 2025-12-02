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
// Forward declarations for SFace helpers
// ==========================================================

bool compute_sface_embedding(
    const FacialAuthConfig &,
    const cv::Mat &face,
    const std::string &,

    cv::Mat &embedding,
    std::string &log
);

bool fa_load_sface_model(
    const std::string &file,
    std::vector<cv::Mat> &embeds
);

bool fa_save_sface_model(
    const FacialAuthConfig &,
    const std::string &profile,
    const std::string &file,
    const std::vector<cv::Mat> &embeds
);


// ==========================================================
// Basic Helpers
// ==========================================================

bool fa_file_exists(const std::string &path)
{
    struct stat st;
    return (::stat(path.c_str(), &st) == 0);
}

static bool is_dir(const std::string &path)
{
    struct stat st;
    if (::stat(path.c_str(), &st) != 0) return false;
    return S_ISDIR(st.st_mode);
}

static void sleep_ms_int(int ms)
{
    if (ms > 0)
        usleep((useconds_t)ms * 1000);
}

// ==========================================================
// Root check
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
// string helpers
// ==========================================================

static std::string trim(const std::string &s)
{
    size_t b = 0, e = s.size();
    while (b < e && std::isspace((unsigned char)s[b])) b++;
    while (e > b && std::isspace((unsigned char)s[e-1])) e--;
    return s.substr(b, e - b);
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
        lineno++;

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

        auto to_bool = [&](const std::string &v, bool &dst){
            std::string low=v;
            std::transform(low.begin(), low.end(), low.begin(), ::tolower);
            dst = (low=="yes" || low=="true" || low=="1");
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
            else if (key == "image_format") cfg.image_format = val;
            else if (key == "recognizer_profile") cfg.recognizer_profile = val;
            else if (key == "detector_profile") cfg.detector_profile = val;
            else if (key == "lbph_threshold") cfg.lbph_threshold = std::stod(val);
            else if (key == "eigen_threshold") cfg.eigen_threshold = std::stod(val);
            else if (key == "fisher_threshold") cfg.fisher_threshold = std::stod(val);
            else if (key == "sface_model") cfg.sface_model = val;
            else if (key == "sface_model_int8") cfg.sface_model_int8 = val;
            else if (key == "haar_model" || key == "detect_haar") cfg.haar_cascade_path = val;
            else if (key == "yunet_model" || key == "detect_yunet_fp32") cfg.yunet_model = val;
            else if (key == "yunet_model_int8" || key == "detect_yunet_int8") cfg.yunet_model_int8 = val;
            else logbuf += "Unknown param '" + key + "'\n";
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
// Paths
// ==========================================================

std::string fa_user_image_dir(const FacialAuthConfig &cfg, const std::string &user)
{
    return (fs::path(cfg.basedir) / "images" / user).string();
}

std::string fa_user_model_path(const FacialAuthConfig &cfg, const std::string &user)
{
    return (fs::path(cfg.basedir) / "models" / (user + ".xml")).string();
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
        }
        else if (cfg.debug) {
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
// DetectorWrapper implementation
// ==========================================================

bool DetectorWrapper::detect(const cv::Mat &frame, cv::Rect &face)
{
    face = cv::Rect();

    if (frame.empty())
        return false;

    int W = frame.cols, H = frame.rows;
    if (debug)
        std::cout << "[DEBUG] detect(): " << W << "x" << H << "\n";

    // ===== HAAR =====
    if (type == DET_HAAR)
    {
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Rect> faces;
        haar.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(30,30));

        if (faces.empty())
            return false;

        face = faces[0];

        if (debug) {
            std::cout << "[DEBUG] HAAR face @ "
            << face.x << "," << face.y << " "
            << face.width << "x" << face.height << "\n";
        }
        return true;
    }

    // ===== YuNet via FaceDetectorYN API =====
    if (type == DET_YUNET && yunet_detector)
    {
        cv::Mat faces;
        yunet_detector->setInputSize(frame.size());
        yunet_detector->detect(frame, faces);

        if (faces.empty())
            return false;

        if (debug)
            std::cout << "[DEBUG] YuNet found " << faces.rows << " candidate(s)\n";

        int bestIdx = -1;
        float bestScore = 0.0f;

        for (int i = 0; i < faces.rows; ++i)
        {
            float score = faces.at<float>(i, 4);
            if (score < 0.6f)
                continue;

            if (bestIdx == -1 || score > bestScore) {
                bestIdx = i;
                bestScore = score;
            }
        }

        if (bestIdx == -1)
            return false;

        cv::Rect r(
            faces.at<float>(bestIdx, 0),
                   faces.at<float>(bestIdx, 1),
                   faces.at<float>(bestIdx, 2),
                   faces.at<float>(bestIdx, 3)
        );

        face = r;

        if (debug) {
            std::cout << "[DEBUG] YuNet best face @ "
            << r.x << "," << r.y
            << " " << r.width << "x" << r.height
            << " score=" << bestScore << "\n";
        }

        return true;
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

    // ===== HAAR =====
    if (low == "haar") {
        if (cfg.haar_cascade_path.empty() || !fa_file_exists(cfg.haar_cascade_path)) {
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

    // ===== YuNet FP32 =====
    if (low == "yunet_fp32") {
        if (cfg.yunet_model.empty() || !fa_file_exists(cfg.yunet_model)) {
            log += "Missing YuNet FP32 model\n";
            return false;
        }

        try {
            det.yunet_detector = cv::FaceDetectorYN::create(
                cfg.yunet_model,
                "",
                cv::Size(cfg.width, cfg.height)
            );

            det.type = DetectorWrapper::DET_YUNET;
            det.debug = cfg.debug;
            det.model_path = cfg.yunet_model;

            log += "Initialized YuNet FP32\n";
            return true;
        }
        catch (...) {
            log += "YuNet FP32 init failed\n";
            return false;
        }
    }

    // ===== YuNet INT8 =====
    if (low == "yunet_int8") {
        if (cfg.yunet_model_int8.empty() || !fa_file_exists(cfg.yunet_model_int8)) {
            log += "Missing YuNet INT8 model\n";
            return false;
        }

        try {
            det.yunet_detector = cv::FaceDetectorYN::create(
                cfg.yunet_model_int8,
                "",
                cv::Size(cfg.width, cfg.height)
            );

            det.type = DetectorWrapper::DET_YUNET;
            det.debug = cfg.debug;
            det.model_path = cfg.yunet_model_int8;

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
        auto name = path.stem().string();      // es: "img_14"
        auto ext  = path.extension().string(); // ".jpg"

        if (ext != "." + format)
            continue;

        if (name.rfind("img_", 0) != 0) // deve iniziare con img_
            continue;

        try {
            int idx = std::stoi(name.substr(4)); // ottiene numero dopo img_
            if (idx > max_idx) max_idx = idx;
        }
        catch (...) {}
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

    // calcolo indice di partenza
    int start_index = 1;
    if (!cfg.force_overwrite) {
        start_index = fa_find_next_image_index(imgdir, img_format);
    } else {
        if (cfg.debug || cfg.verbose)
            std::cout << "[INFO] Force mode: starting index at 1\n";
    }

    if (cfg.debug || cfg.verbose) {
        std::cout << "[INFO] Saving captured images to: " << imgdir << "\n";
        std::cout << "[INFO] Starting index: " << start_index << "\n";
    }

    cv::VideoCapture cap;
    if (!open_camera(cap, cfg, log)) {
        log += "[ERROR] Cannot open camera.\n";
        return false;
    }

    DetectorWrapper detector;
    if (!init_detector(cfg, detector, log)) {
        log += "[ERROR] Cannot initialize detector (profile=" + cfg.detector_profile + ").\n";
        return false;
    }

    int saved = 0;

    for (int i = 0; i < cfg.frames; ++i)
    {
        cv::Mat frame;
        if (!capture_frame(cap, frame, cfg, log)) {
            log += "[ERROR] Invalid frame from camera.\n";
            break;
        }

        cv::Rect face;
        if (!detector.detect(frame, face))
            continue;   // salto se non c'è faccia

            int idx = start_index + saved;
        std::string outfile = imgdir + "/img_" + std::to_string(idx) + "." + img_format;

        if (!cv::imwrite(outfile, frame))
        {
            log += "[ERROR] Cannot save image: " + outfile + "\n";
        }
        else
        {
            ++saved;

            if (cfg.debug || cfg.verbose)
            {
                std::cout << "[DEBUG] (" << saved << "/" << cfg.frames
                << ") Saved: " << outfile << "\n";
            }
            else
            {
                // output minimo sempre visibile
                std::cout << "[INFO] Saved: " << outfile << "\n";
            }
        }

        if (cfg.sleep_ms > 0)
            sleep_ms_int(cfg.sleep_ms);
    }

    if (saved == 0) {
        log += "[WARN] No images saved: no face detected.\n";
        return false;
    }

    log += "[INFO] Capture complete. Images saved: " + std::to_string(saved) + "\n";
    return true;
}



// ==========================================================
// Training helpers (classic LBPH/Eigen/Fisher)
// ==========================================================

static bool train_classic(
    const std::string &,
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
            log += "No face detected in: " + fn + "\n";
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

    cv::Ptr<cv::face::FaceRecognizer> rec;

    std::string mlow = method;
    std::transform(mlow.begin(), mlow.end(), mlow.begin(),
                   [](unsigned char c){ return std::tolower(c); });

    if (mlow == "lbph") {
        rec = cv::face::LBPHFaceRecognizer::create(
            1, 8, 8, 8, cfg.lbph_threshold
        );
    }
    else if (mlow == "eigen") {
        rec = cv::face::EigenFaceRecognizer::create(
            cfg.eigen_components, cfg.eigen_threshold
        );
    }
    else if (mlow == "fisher") {
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

    try {
        rec->train(faces, labels);
    }
    catch (const std::exception &e) {
        log += std::string("Training failed: ") + e.what() + "\n";
        return false;
    }

    if (fa_file_exists(model_path) && !force_overwrite) {
        log += "Model exists (use --force to overwrite): " + model_path + "\n";
        return false;
    }

    try {
        rec->save(model_path);
    }
    catch (const std::exception &e) {
        log += "Failed to save model: ";
        log += e.what();
        log += "\n";
        return false;
    }

    log += "Classic model saved: " + model_path + "\n";
    return true;
}
// ==========================================================
// Cosine similarity helper
// ==========================================================
static double cosine_similarity(const cv::Mat &a, const cv::Mat &b)
{
    if (a.empty() || b.empty())
        return 0.0;

    double dot = a.dot(b);
    double na  = cv::norm(a);
    double nb  = cv::norm(b);

    if (na <= 0.0 || nb <= 0.0)
        return 0.0;

    return dot / (na * nb);
}


// ==========================================================
// Public API: Train user model
// ==========================================================
bool fa_train_user(
    const std::string &user,
    const FacialAuthConfig &cfg,
    std::string &log
)
{
    std::string imgdir  = fa_user_image_dir(cfg, user);
    std::string model   = fa_user_model_path(cfg, user);
    std::string method  = cfg.training_method;

    if (!is_dir(imgdir)) {
        log += "Training aborted: no images for user: " + user + "\n";
        return false;
    }

    std::string mlow = method;
    std::transform(mlow.begin(), mlow.end(), mlow.begin(),
                   [](unsigned char c){ return std::tolower(c); });

    bool classic = (mlow == "lbph" || mlow == "eigen" ||
    mlow == "fisher" || mlow.empty());

    if (classic) {
        return train_classic(
            user, cfg, mlow, imgdir, model,
            cfg.force_overwrite, log
        );
    }

    if (mlow == "sface") {
        DetectorWrapper det;
        if (!init_detector(cfg, det, log)) {
            log += "Cannot init detector for SFace training\n";
            return false;
        }

        std::vector<cv::Mat> gallery;

        for (auto &p : fs::directory_iterator(imgdir)) {
            if (!p.is_regular_file()) continue;
            auto ext = p.path().extension().string();
            if (ext != ".jpg" && ext != ".png") continue;

            cv::Mat img = cv::imread(p.path().string());
            if (img.empty()) continue;

            cv::Rect face;
            if (!det.detect(img, face)) continue;

            cv::Mat roi = img(face).clone();
            cv::resize(roi, roi, cv::Size(112,112));

            std::string tmp;
            cv::Mat emb;

            if (compute_sface_embedding(cfg, roi, cfg.recognizer_profile, emb, tmp))
                gallery.push_back(emb);
        }

        if (gallery.empty()) {
            log += "SFace training failed: no embeddings\n";
            return false;
        }

        if (!fa_save_sface_model(cfg, cfg.recognizer_profile, model, gallery)) {
            log += "Failed to save SFace model: " + model + "\n";
            return false;
        }

        log += "SFace model saved: " + model + "\n";
        return true;
    }

    log += "Unknown training method: " + method + "\n";
    return false;
}


// ==========================================================
// Authentication Test Function
// ==========================================================
bool fa_test_user(
    const std::string &,
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

    if (!fa_file_exists(modelPath)) {
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
        if (threshold_override >= 0.0)
            thr = threshold_override;
        else {
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
        rec = cv::face::LBPHFaceRecognizer::create(
            1, 8, 8, 8, cfg.lbph_threshold
        );
        rec->read(modelPath);
    }
    catch (const cv::Exception &e) {
        log += "Failed to load classic recognizer: ";
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

// ==========================================================
// Remove user data (models + images)
// ==========================================================
bool fa_delete_user(
    const std::string &user,
    const FacialAuthConfig &cfg,
    std::string &log
)
{
    std::string imgdir  = fa_user_image_dir(cfg, user);
    std::string model   = fa_user_model_path(cfg, user);

    bool removed_any = false;

    try {
        if (fa_file_exists(model)) {
            fs::remove(model);
            log += "Deleted model: " + model + "\n";
            removed_any = true;
        }
    }
    catch (const std::exception &e) {
        log += "Error deleting model file: ";
        log += e.what();
        log += "\n";
    }

    try {
        if (fs::exists(imgdir)) {
            fs::remove_all(imgdir);
            log += "Deleted image directory: " + imgdir + "\n";
            removed_any = true;
        }
    }
    catch (const std::exception &e) {
        log += "Error deleting image directory: ";
        log += e.what();
        log += "\n";
    }

    return removed_any;
}


// ==========================================================
// Enumerate registered users
// ==========================================================
bool fa_list_users(
    const FacialAuthConfig &cfg,
    std::vector<std::string> &users,
    std::string &log
)
{
    users.clear();

    std::string base = cfg.basedir.empty()
    ? "/var/lib/pam_facial_auth"
    : cfg.basedir;

    fs::path model_dir = fs::path(base) / "models";

    if (!fs::exists(model_dir)) {
        log += "No model dir exists: " + model_dir.string() + "\n";
        return true;
    }

    try {
        for (auto &p : fs::directory_iterator(model_dir)) {
            if (!p.is_regular_file()) continue;

            auto path = p.path();
            if (path.extension() != ".xml") continue;

            users.push_back(path.stem().string());
        }
    }
    catch (const std::exception &e) {
        log += "Error listing users: ";
        log += e.what();
        log += "\n";
        return false;
    }

    return true;
}


// ==========================================================
// Check if user exists
// ==========================================================
bool fa_user_exists(
    const std::string &user,
    const FacialAuthConfig &cfg
)
{
    return fa_file_exists(fa_user_model_path(cfg, user));
}


// ==========================================================
// Debug dump
// ==========================================================
void fa_dump_config(
    const FacialAuthConfig &cfg,
    std::ostream &os
)
{
    os << "[CONFIG] basedir=" << cfg.basedir << "\n"
    << "[CONFIG] device="  << cfg.device  << "\n"
    << "[CONFIG] resolution=" << cfg.width << "x" << cfg.height << "\n"
    << "[CONFIG] frames="     << cfg.frames << "\n"
    << "[CONFIG] detector_profile=" << cfg.detector_profile << "\n"
    << "[CONFIG] recognizer_profile=" << cfg.recognizer_profile << "\n"
    << "\n";
}

// ==========================================================
// SFACE embedding + model helpers
// ==========================================================

bool compute_sface_embedding(
    const FacialAuthConfig &,
    const cv::Mat &face,
    const std::string &,

    cv::Mat &embedding,
    std::string &log
)
{
    if (face.empty()) {
        log += "[SFACE] ERROR: empty input image\n";
        return false;
    }

    try {
        cv::Mat gray;
        if (face.channels() == 3)
            cv::cvtColor(face, gray, cv::COLOR_BGR2GRAY);
        else
            gray = face.clone();

        gray = gray.reshape(1, 1);
        gray.convertTo(embedding, CV_32F);
        cv::normalize(embedding, embedding);
        return true;
    }
    catch (const std::exception &e) {
        log += "[SFACE] EXCEPTION: ";
        log += e.what();
        log += "\n";
        return false;
    }
}

bool fa_save_sface_model(
    const FacialAuthConfig &,
    const std::string &profile,
    const std::string &file,
    const std::vector<cv::Mat> &embeds
)
{
    try {
        cv::FileStorage fs(file, cv::FileStorage::WRITE);
        if (!fs.isOpened())
            return false;

        fs << "profile" << profile;
        fs << "embeddings" << "[";

        for (auto &e : embeds)
            fs << e;

        fs << "]";
        return true;
    }
    catch (...) {
        return false;
    }
}


bool fa_load_sface_model(
    const std::string &file,
    std::vector<cv::Mat> &embeds
)
{
    try {
        cv::FileStorage fs(file, cv::FileStorage::READ);
        if (!fs.isOpened())
            return false;

        embeds.clear();
        cv::FileNode n = fs["embeddings"];
        if (n.type() != cv::FileNode::SEQ)
            return false;

        for (auto it = n.begin(); it != n.end(); ++it) {
            cv::Mat m;
            (*it) >> m;
            embeds.push_back(m);
        }
        return true;
    }
    catch (...) {
        return false;
    }
}


// ==========================================================
// Public C API export for PAM / CLI wrappers
// ==========================================================
extern "C" {

    bool facialauth_load_config(FacialAuthConfig *cfg, const char *path, char *logbuf, int logbuflen)
    {
        if (!cfg || !logbuf) return false;
        std::string log;
        FacialAuthConfig c;
        if (!fa_load_config(c, log, path ? path : "")) {
            strncpy(logbuf, log.c_str(), logbuflen-1);
            logbuf[logbuflen-1] = 0;
            return false;
        }
        *cfg = c;
        strncpy(logbuf, log.c_str(), logbuflen-1);
        logbuf[logbuflen-1] = 0;
        return true;
    }

    bool facialauth_train(const char *user, const FacialAuthConfig *cfg, char *logbuf, int logbuflen)
    {
        if (!user || !cfg || !logbuf) return false;
        std::string log;
        bool ok = fa_train_user(user, *cfg, log);
        strncpy(logbuf, log.c_str(), logbuflen-1);
        logbuf[logbuflen-1] = 0;
        return ok;
    }

    bool facialauth_test(
        const char *user,
        const FacialAuthConfig *cfg,
        const char *modelPath,
        double *best_conf,
        int *best_label,
        char *logbuf,
        int logbuflen
    )
    {
        if (!user || !cfg || !modelPath || !logbuf || !best_conf || !best_label)
            return false;

        std::string log;
        double bc = 0.0;
        int bl = -1;
        bool ok = fa_test_user(user, *cfg, modelPath, bc, bl, log);

        *best_conf  = bc;
        *best_label = bl;

        strncpy(logbuf, log.c_str(), logbuflen-1);
        logbuf[logbuflen-1] = 0;
        return ok;
    }

    bool facialauth_delete(
        const char *user,
        const FacialAuthConfig *cfg,
        char *logbuf,
        int logbuflen
    )
    {
        if (!user || !cfg || !logbuf) return false;

        std::string log;
        bool ok = fa_delete_user(user, *cfg, log);

        strncpy(logbuf, log.c_str(), logbuflen-1);
        logbuf[logbuflen-1] = 0;
        return ok;
    }


} // extern "C"
