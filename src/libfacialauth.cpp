#include "../include/libfacialauth.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/face.hpp>
#include <opencv2/highgui.hpp>



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
#include <unistd.h>   // per usleep

#include <filesystem>
namespace fs = std::filesystem;

using std::string;
using std::cerr;
using std::cout;
using std::endl;

bool fa_check_root(const std::string &tool_name)
{
    if (::geteuid() != 0) {
        std::cerr << "[" << tool_name << "] must be run as root.\n";
        return false;
    }
    return true;
}



// ==========================================================
// Helpers
// ==========================================================

static bool file_exists(const string &path)
{
    struct stat st;
    return ::stat(path.c_str(), &st) == 0;
}

static bool is_dir(const string &path)
{
    struct stat st;
    if (::stat(path.c_str(), &st) != 0) return false;
    return S_ISDIR(st.st_mode);
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
// Config loader
// ==========================================================

static inline string trim(const string &s)
{
    size_t b = 0;
    while (b < s.size() && std::isspace((unsigned char)s[b])) ++b;
    size_t e = s.size();
    while (e > b && std::isspace((unsigned char)s[e - 1])) --e;
    return s.substr(b, e - b);
}

static inline bool starts_with(const string &s, const string &prefix)
{
    if (s.size() < prefix.size()) return false;
    return std::equal(prefix.begin(), prefix.end(), s.begin());
}

bool fa_load_config(
    FacialAuthConfig &cfg,
    std::string &logbuf,
    const std::string &path
)
{
    // reset con valori di default della struct
    cfg = FacialAuthConfig();
    logbuf.clear();

    //
    // Determina il path del file di configurazione:
    //  - se la CLI ha passato una stringa vuota → usa default
    //  - altrimenti usa quello specificato
    //
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

        auto to_bool = [&](const std::string &v, bool &dest) {
            std::string low = v;
            std::transform(low.begin(), low.end(), low.begin(),
                           [](unsigned char c){ return std::tolower(c); });
            dest = (low == "yes" || low == "true" || low == "1");
        };

        try {
            if (key == "basedir") {
                cfg.basedir = val;
            } else if (key == "device") {
                cfg.device = val;
            } else if (key == "fallback_device") {
                to_bool(val, cfg.fallback_device);
            } else if (key == "width") {
                cfg.width = std::stoi(val);
            } else if (key == "height") {
                cfg.height = std::stoi(val);
            } else if (key == "frames") {
                cfg.frames = std::stoi(val);
            } else if (key == "sleep_ms") {
                cfg.sleep_ms = std::stoi(val);
            } else if (key == "debug") {
                to_bool(val, cfg.debug);
            } else if (key == "verbose") {
                to_bool(val, cfg.verbose);
            } else if (key == "nogui") {
                to_bool(val, cfg.nogui);
            } else if (key == "training_method") {
                cfg.training_method = val;
            } else if (key == "force_overwrite") {
                to_bool(val, cfg.force_overwrite);
            } else if (key == "ignore_failure") {
                to_bool(val, cfg.ignore_failure);
            } else if (key == "save_failed_images") {
                to_bool(val, cfg.save_failed_images);
            } else if (key == "image_format") {
                cfg.image_format = val;

                // Detector / recognizer profiles
            } else if (key == "detector_profile") {
                cfg.detector_profile = val;
            } else if (key == "recognizer_profile") {
                cfg.recognizer_profile = val;

                // Thresholds
            } else if (key == "lbph_threshold") {
                cfg.lbph_threshold = std::stod(val);
            } else if (key == "eigen_threshold") {
                cfg.eigen_threshold = std::stod(val);
            } else if (key == "fisher_threshold") {
                cfg.fisher_threshold = std::stod(val);

            } else if (key == "eigen_components") {
                cfg.eigen_components = std::stoi(val);
            } else if (key == "fisher_components") {
                cfg.fisher_components = std::stoi(val);

                // SFace thresholds
            } else if (key == "sface_threshold") {
                cfg.sface_threshold      = std::stod(val);
                cfg.sface_fp32_threshold = cfg.sface_threshold;
                cfg.sface_int8_threshold = cfg.sface_threshold;
            } else if (key == "sface_fp32_threshold") {
                cfg.sface_fp32_threshold = std::stod(val);
            } else if (key == "sface_int8_threshold") {
                cfg.sface_int8_threshold = std::stod(val);

                // DNN backend/target
            } else if (key == "dnn_backend") {
                cfg.dnn_backend = val;
            } else if (key == "dnn_target") {
                cfg.dnn_target = val;

                // Legacy fields
            } else if (key == "model_path") {
                cfg.model_path = val;
            } else if (key == "haar_cascade_path") {
                cfg.haar_cascade_path = val;

                // Dynamic detectors detect_*
            } else if (starts_with(key, "detect_")) {
                std::string subkey = key.substr(7);
                cfg.detector_models[subkey] = val;

                if (subkey == "haar" || subkey == "haar_model") {
                    cfg.haar_cascade_path = val;
                    cfg.detector_models["haar"] = val;
                }
                else if (subkey == "yunet_fp32" || subkey == "yunet_model_fp32") {
                    cfg.yunet_model = val;
                    cfg.detector_models["yunet_fp32"] = val;
                }
                else if (subkey == "yunet_int8" || subkey == "yunet_model_int8") {
                    cfg.yunet_model_int8 = val;
                    cfg.detector_models["yunet_int8"] = val;
                }

                // Dynamic recognizers recognize_*
            } else if (starts_with(key, "recognize_")) {
                std::string subkey = key.substr(10);
                cfg.recognizer_models[subkey] = val;

                if (subkey == "sface_fp32" || subkey == "sface_model_fp32") {
                    cfg.sface_model = val;
                    cfg.recognizer_models["sface_fp32"] = val;
                }
                else if (subkey == "sface_int8" || subkey == "sface_model_int8") {
                    cfg.sface_model_int8 = val;
                    cfg.recognizer_models["sface_int8"] = val;
                }

                // Backward compatibility keys
            } else if (key == "yunet_model") {
                cfg.yunet_model = val;
                cfg.detector_models["yunet_fp32"] = val;
            } else if (key == "yunet_model_int8") {
                cfg.yunet_model_int8 = val;
                cfg.detector_models["yunet_int8"] = val;
            } else if (key == "haar_model") {
                cfg.haar_cascade_path = val;
                cfg.detector_models["haar"] = val;
            } else if (key == "sface_model") {
                cfg.sface_model = val;
                cfg.recognizer_models["sface_fp32"] = val;
            } else if (key == "sface_model_int8") {
                cfg.sface_model_int8 = val;
                cfg.recognizer_models["sface_int8"] = val;
            }

            else {
                logbuf += "Unknown key at line " + std::to_string(lineno) +
                ": '" + key + "'\n";
            }
        }
        catch (const std::exception &e) {
            logbuf += "Error parsing line " + std::to_string(lineno) +
            " ('" + key + "'): " + e.what() + "\n";
        }
    }

    f.close();

    // Se basedir è vuoto, imposta default
    if (cfg.basedir.empty())
        cfg.basedir = "/var/lib/pam_facial_auth";

    return true;
}

// ==========================================================
// Paths
// ==========================================================

std::string fa_user_image_dir(
    const FacialAuthConfig &cfg,
    const std::string &user
)
{
    fs::path base(cfg.basedir.empty() ? "/var/lib/pam_facial_auth" : cfg.basedir);
    fs::path p = base / "images" / user;
    return p.string();
}

std::string fa_user_model_path(
    const FacialAuthConfig &cfg,
    const std::string &user
)
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
            if (cfg.debug) {
                log += "Opened camera: " + d + "\n";
            }
            return true;
        } else {
            if (cfg.debug) {
                log += "Failed to open camera: " + d + "\n";
            }
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
// SFace model save/load helpers (XML with metadata)
// ==========================================================

static bool fa_save_sface_model(const FacialAuthConfig &cfg,
                                const std::string &profile,
                                const std::string &file,
                                const std::vector<cv::Mat> &embeds)
{
    try {
        ensure_dirs(fs::path(file).parent_path().string());
        cv::FileStorage fs(file, cv::FileStorage::WRITE);
        if (!fs.isOpened()) return false;

        fs << "type" << "sface";
        fs << "version" << 1;

        fs << "recognizer_profile" << profile;
        fs << "detector_profile"   << cfg.detector_profile;
        fs << "dnn_backend"        << cfg.dnn_backend;
        fs << "dnn_target"         << cfg.dnn_target;
        fs << "sface_fp32_threshold" << cfg.sface_fp32_threshold;
        fs << "sface_int8_threshold" << cfg.sface_int8_threshold;
        fs << "width"              << cfg.width;
        fs << "height"             << cfg.height;
        fs << "frames"             << cfg.frames;

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
        if (type != "sface") {
            return false;
        }

        cv::FileNode emb = fs["embeddings"];
        if (emb.empty() || emb.type() != cv::FileNode::SEQ) {
            return false;
        }

        for (auto it = emb.begin(); it != emb.end(); ++it) {
            cv::Mat m;
            (*it) >> m;
            if (!m.empty())
                embeds.push_back(m);
        }

        fs.release();
        return !embeds.empty();
    } catch (...) {
        return false;
    }
}

// ==========================================================
// DNN backend/target helpers
// ==========================================================

static int parse_dnn_backend(const std::string &b)
{
    std::string low = b;
    std::transform(low.begin(), low.end(), low.begin(),
                   [](unsigned char c){ return std::tolower(c); });

    if (low == "cpu")        return cv::dnn::DNN_BACKEND_OPENCV;
    if (low == "cuda")       return cv::dnn::DNN_BACKEND_CUDA;
    if (low == "cuda_fp16")  return cv::dnn::DNN_BACKEND_CUDA;
    if (low == "opencl")     return cv::dnn::DNN_BACKEND_OPENCV;
    return cv::dnn::DNN_BACKEND_DEFAULT;
}

static int parse_dnn_target(const std::string &t)
{
    std::string low = t;
    std::transform(low.begin(), low.end(), low.begin(),
                   [](unsigned char c){ return std::tolower(c); });

    if (low == "cpu")        return cv::dnn::DNN_TARGET_CPU;
    if (low == "cuda")       return cv::dnn::DNN_TARGET_CUDA;
    if (low == "cuda_fp16")  return cv::dnn::DNN_TARGET_CUDA_FP16;
    if (low == "opencl")     return cv::dnn::DNN_TARGET_OPENCL;
    return cv::dnn::DNN_TARGET_CPU;
}

// ==========================================================
// Detector wrapper (Haar / YuNet)
// ==========================================================

struct DetectorWrapper {
    enum Type {
        DET_NONE,
        DET_HAAR,
        DET_YUNET
    } type = DET_NONE;

    mutable cv::CascadeClassifier haar;
    cv::Ptr<cv::dnn::Net> yunet;
    cv::Size input_size = cv::Size(320, 320);

    std::string model_path;

    bool detect(const cv::Mat &frame, cv::Rect &face) const;
};

bool DetectorWrapper::detect(const cv::Mat &frame, cv::Rect &face) const
{
    face = cv::Rect();

    auto dbg = [&](const std::string &msg){
        if (debug)
            std::cerr << "[DETECT] " << msg << "\n";
    };

    if (frame.empty()) {
        dbg("Frame vuoto");
        return false;
    }

    // --- HAAR ---
    if (type == DET_HAAR) {
        std::vector<cv::Rect> faces;
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        haar.detectMultiScale(
            gray, faces, 1.1, 3,
            0 | cv::CASCADE_SCALE_IMAGE,
            cv::Size(30, 30)
        );

        if (!faces.empty()) {
            face = faces[0];
            dbg("Volto rilevato (HAAR)");
            return true;
        }

        dbg("Nessun volto (HAAR)");
        return false;
    }

    // --- YUNET ---
    if (type == DET_YUNET && yunet) {

        cv::Mat blob = cv::dnn::blobFromImage(
            frame, 1.0, input_size,
            cv::Scalar(104,117,123), true, false
        );

        yunet->setInput(blob);
        cv::Mat out = yunet->forward();

        if (out.empty() || out.dims != 3) {
            dbg("YuNet output non valido");
            return false;
        }

        float best_score = 0.f;
        cv::Rect best_rect;

        const int num = out.size[1];
        float *data = (float*)out.data;

        for (int i=0; i < num; i++) {

            float x = data[i*15 + 0];
            float y = data[i*15 + 1];
            float w = data[i*15 + 2];
            float h = data[i*15 + 3];
            float score = data[i*15 + 4];

            if (score < 0.75f) {
                dbg("Scarto: score basso " + std::to_string(score));
                continue;
            }

            if (w < 40 || h < 40) {
                dbg("Scarto: box troppo piccolo " + std::to_string(w) + "x" + std::to_string(h));
                continue;
            }

            if (score > best_score) {
                best_score = score;
                best_rect = cv::Rect((int)x, (int)y, (int)w, (int)h);
            }
        }

        if (best_score <= 0.f) {
            dbg("Nessun volto valido (YuNet)");
            return false;
        }

        face = best_rect & cv::Rect(0,0,frame.cols, frame.rows);
        dbg("Volto rilevato YuNet: "
        "x=" + std::to_string(face.x) +
        " y=" + std::to_string(face.y) +
        " w=" + std::to_string(face.width) +
        " h=" + std::to_string(face.height) +
        " score=" + std::to_string(best_score));

        return true;
    }

    dbg("Detector non inizializzato");
    return false;
}


// ==========================================================
// Recognizer wrapper (classic LBPH/Eigen/Fisher)
// ==========================================================

class FaceRecWrapper {
public:
    enum Type {
        TYPE_NONE,
        TYPE_LBPH,
        TYPE_EIGEN,
        TYPE_FISHER
    };

    FaceRecWrapper() : type_(TYPE_NONE) {}

    bool Create(const std::string &method,
                double lbph_threshold,
                double eigen_threshold,
                double fisher_threshold,
                int eigen_components,
                int fisher_components);

    bool Train(const std::vector<cv::Mat> &faces,
               const std::vector<int> &labels,
               std::string &err);

    bool Save(const std::string &file, std::string &err) const;

    bool Load(const std::string &file, std::string &err);

    bool Predict(const cv::Mat &face,
                 int &label,
                 double &confidence,
                 std::string &err) const;

                 Type type() const { return type_; }

private:
    Type type_;
    cv::Ptr<cv::face::FaceRecognizer> recognizer_;
    double lbph_threshold_;
    double eigen_threshold_;
    double fisher_threshold_;
};

bool FaceRecWrapper::Create(const std::string &method,
                            double lbph_threshold,
                            double eigen_threshold,
                            double fisher_threshold,
                            int eigen_components,
                            int fisher_components)
{
    std::string m = method;
    std::transform(m.begin(), m.end(), m.begin(),
                   [](unsigned char c){ return std::tolower(c); });

    if (m == "lbph") {
        type_ = TYPE_LBPH;
        recognizer_ = cv::face::LBPHFaceRecognizer::create();
        lbph_threshold_ = lbph_threshold;
        recognizer_->setThreshold(lbph_threshold_);
        return true;
    } else if (m == "eigen") {
        type_ = TYPE_EIGEN;
        recognizer_ = cv::face::EigenFaceRecognizer::create(eigen_components, eigen_threshold);
        eigen_threshold_ = eigen_threshold;
        return true;
    } else if (m == "fisher") {
        type_ = TYPE_FISHER;
        recognizer_ = cv::face::FisherFaceRecognizer::create(fisher_components, fisher_threshold);
        fisher_threshold_ = fisher_threshold;
        return true;
    }

    type_ = TYPE_NONE;
    return false;
}

bool FaceRecWrapper::Train(const std::vector<cv::Mat> &faces,
                           const std::vector<int> &labels,
                           std::string &err)
{
    if (!recognizer_) {
        err = "FaceRecWrapper: recognizer not created.\n";
        return false;
    }
    try {
        recognizer_->train(faces, labels);
        return true;
    } catch (const std::exception &e) {
        err = std::string("FaceRecWrapper::Train error: ") + e.what() + "\n";
        return false;
    }
}

bool FaceRecWrapper::Save(const std::string &file, std::string &err) const
{
    if (!recognizer_) {
        err = "FaceRecWrapper: recognizer not created.\n";
        return false;
    }
    try {
        ensure_dirs(fs::path(file).parent_path().string());
        recognizer_->write(file);
        return true;
    } catch (const std::exception &e) {
        err = std::string("FaceRecWrapper::Save error: ") + e.what() + "\n";
        return false;
    }
}

bool FaceRecWrapper::Load(const std::string &file, std::string &err)
{
    try {
        recognizer_ = cv::face::LBPHFaceRecognizer::create();
        recognizer_->read(file);
        type_ = TYPE_LBPH;  // best guess
        return true;
    } catch (const std::exception &e) {
        err = std::string("FaceRecWrapper::Load error: ") + e.what() + "\n";
        return false;
    }
}

bool FaceRecWrapper::Predict(const cv::Mat &face,
                             int &label,
                             double &confidence,
                             std::string &err) const
                             {
                                 if (!recognizer_) {
                                     err = "FaceRecWrapper: recognizer not created.\n";
                                     return false;
                                 }
                                 try {
                                     recognizer_->predict(face, label, confidence);
                                     return true;
                                 } catch (const std::exception &e) {
                                     err = std::string("FaceRecWrapper::Predict error: ") + e.what() + "\n";
                                     return false;
                                 }
                             }

                             // ==========================================================
                             // SFace embedding helper
                             // ==========================================================

                             static bool resolve_sface_model(
                                 const FacialAuthConfig &cfg,
                                 const std::string &profile,
                                 std::string &model_path,
                                 std::string &used_profile
                             )
                             {
                                 used_profile = profile;
                                 std::string prof = profile;
                                 if (prof.empty())
                                     prof = cfg.recognizer_profile;

                                 std::transform(prof.begin(), prof.end(), prof.begin(),
                                                [](unsigned char c){ return std::tolower(c); });

                                 auto it = cfg.recognizer_models.find(prof);
                                 if (it != cfg.recognizer_models.end()) {
                                     model_path = it->second;
                                     used_profile = prof;
                                     return true;
                                 }

                                 if (prof == "sface_fp32" || prof == "sface") {
                                     if (!cfg.sface_model.empty()) {
                                         model_path = cfg.sface_model;
                                         used_profile = "sface_fp32";
                                         return true;
                                     }
                                 } else if (prof == "sface_int8") {
                                     if (!cfg.sface_model_int8.empty()) {
                                         model_path = cfg.sface_model_int8;
                                         used_profile = "sface_int8";
                                         return true;
                                     }
                                 }

                                 if (!cfg.sface_model.empty()) {
                                     model_path = cfg.sface_model;
                                     used_profile = "sface_fp32";
                                     return true;
                                 }
                                 if (!cfg.sface_model_int8.empty()) {
                                     model_path = cfg.sface_model_int8;
                                     used_profile = "sface_int8";
                                     return true;
                                 }

                                 return false;
                             }

                             static bool compute_sface_embedding(
                                 const FacialAuthConfig &cfg,
                                 const cv::Mat &face,
                                 const std::string &profile,
                                 cv::Mat &embedding,
                                 std::string &log
                             )
                             {
                                 std::string model_path;
                                 std::string used_profile;
                                 if (!resolve_sface_model(cfg, profile, model_path, used_profile)) {
                                     log += "SFace model not configured for profile '" + profile + "'.\n";
                                     return false;
                                 }

                                 if (!file_exists(model_path)) {
                                     log += "SFace model file not found: " + model_path + "\n";
                                     return false;
                                 }

                                 try {
                                     cv::dnn::Net net = cv::dnn::readNetFromONNX(model_path);
                                     if (net.empty()) {
                                         log += "Failed to load SFace ONNX model: " + model_path + "\n";
                                         return false;
                                     }

                                     int backend = parse_dnn_backend(cfg.dnn_backend);
                                     int target  = parse_dnn_target(cfg.dnn_target);

                                     net.setPreferableBackend(backend);
                                     net.setPreferableTarget(target);

                                     cv::Mat blob = cv::dnn::blobFromImage(
                                         face, 1.0 / 255.0,
                                         cv::Size(112, 112),
                                                                           cv::Scalar(0,0,0),
                                                                           true, false
                                     );
                                     net.setInput(blob);
                                     cv::Mat out = net.forward();

                                     if (out.empty()) {
                                         log += "SFace forward() produced empty output.\n";
                                         return false;
                                     }

                                     cv::Mat e;
                                     out.reshape(1, 1).convertTo(e, CV_32F);
                                     double norm = cv::norm(e);
                                     if (norm > 0.0)
                                         e /= norm;

                                     embedding = e;
                                     if (cfg.debug) {
                                         log += "Computed SFace embedding using model '" +
                                         model_path + "' profile='" + used_profile + "'.\n";
                                     }

                                     return true;
                                 } catch (const std::exception &ex) {
                                     log += std::string("Exception in compute_sface_embedding: ") + ex.what() + "\n";
                                     return false;
                                 }
                             }

                             // ==========================================================
                             // Detector init
                             // ==========================================================

                             static bool init_detector(const FacialAuthConfig &cfg,
                                                       DetectorWrapper &det,
                                                       std::string &log)
                             {
                                 det = DetectorWrapper();

                                 std::string profile = cfg.detector_profile;
                                 if (profile.empty())
                                     profile = "auto";

                                 std::string low = profile;
                                 std::transform(low.begin(), low.end(), low.begin(),
                                                [](unsigned char c){ return std::tolower(c); });

                                 if (low == "auto") {
                                     if (cfg.detector_models.count("yunet_fp32") &&
                                         file_exists(cfg.detector_models.at("yunet_fp32"))) {
                                         std::string path = cfg.detector_models.at("yunet_fp32");
                                     try {
                                         det.yunet = cv::makePtr<cv::dnn::Net>(cv::dnn::readNetFromONNX(path));
                                         det.type = DetectorWrapper::DET_YUNET;
                                         det.model_path = path;

                                         int backend = parse_dnn_backend(cfg.dnn_backend);
                                         int target  = parse_dnn_target(cfg.dnn_target);
                                         det.yunet->setPreferableBackend(backend);
                                         det.yunet->setPreferableTarget(target);

                                         if (cfg.debug) {
                                             log += "Using YuNet FP32 detector: " + path + "\n";
                                         }
                                         return true;
                                     } catch (const std::exception &e) {
                                         log += std::string("Failed to init YuNet FP32: ") + e.what() + "\n";
                                     }
                                         }

                                         if (cfg.detector_models.count("yunet_int8") &&
                                             file_exists(cfg.detector_models.at("yunet_int8"))) {
                                             std::string path = cfg.detector_models.at("yunet_int8");
                                         try {
                                             det.yunet = cv::makePtr<cv::dnn::Net>(cv::dnn::readNetFromONNX(path));
                                             det.type = DetectorWrapper::DET_YUNET;
                                             det.model_path = path;

                                             int backend = parse_dnn_backend(cfg.dnn_backend);
                                             int target  = parse_dnn_target(cfg.dnn_target);
                                             det.yunet->setPreferableBackend(backend);
                                             det.yunet->setPreferableTarget(target);

                                             if (cfg.debug) {
                                                 log += "Using YuNet INT8 detector: " + path + "\n";
                                             }
                                             return true;
                                         } catch (const std::exception &e) {
                                             log += std::string("Failed to init YuNet INT8: ") + e.what() + "\n";
                                         }
                                             }

                                             if (cfg.detector_models.count("haar") &&
                                                 file_exists(cfg.detector_models.at("haar"))) {
                                                 std::string path = cfg.detector_models.at("haar");
                                             if (det.haar.load(path)) {
                                                 det.type = DetectorWrapper::DET_HAAR;
                                                 det.model_path = path;
                                                 if (cfg.debug) {
                                                     log += "Using Haar detector: " + path + "\n";
                                                 }
                                                 return true;
                                             } else {
                                                 log += "Failed to load Haar cascade: " + path + "\n";
                                             }
                                                 }

                                                 log += "No suitable detector found in auto mode.\n";
                                                 return false;
                                 }

                                 // Espliciti
                                 if (low == "haar") {
                                     std::string path;
                                     auto it = cfg.detector_models.find("haar");
                                     if (it != cfg.detector_models.end())
                                         path = it->second;
                                     else
                                         path = cfg.haar_cascade_path;

                                     if (path.empty()) {
                                         log += "Haar detector requested but haar model path is empty.\n";
                                         return false;
                                     }
                                     if (!file_exists(path)) {
                                         log += "Haar detector file not found: " + path + "\n";
                                         return false;
                                     }
                                     if (!det.haar.load(path)) {
                                         log += "Failed to load Haar cascade: " + path + "\n";
                                         return false;
                                     }
                                     det.type = DetectorWrapper::DET_HAAR;
                                     det.model_path = path;
                                     if (cfg.debug) {
                                         log += "Using Haar detector: " + path + "\n";
                                     }
                                     return true;
                                 }

                                 if (low == "yunet" || low == "yunet_fp32") {
                                     std::string path;
                                     auto it = cfg.detector_models.find("yunet_fp32");
                                     if (it != cfg.detector_models.end())
                                         path = it->second;
                                     else
                                         path = cfg.yunet_model;

                                     if (path.empty() || !file_exists(path)) {
                                         log += "YuNet FP32 file not found.\n";
                                         return false;
                                     }
                                     try {
                                         det.yunet = cv::makePtr<cv::dnn::Net>(cv::dnn::readNetFromONNX(path));
                                         det.type = DetectorWrapper::DET_YUNET;
                                         det.model_path = path;

                                         int backend = parse_dnn_backend(cfg.dnn_backend);
                                         int target  = parse_dnn_target(cfg.dnn_target);
                                         det.yunet->setPreferableBackend(backend);
                                         det.yunet->setPreferableTarget(target);

                                         if (cfg.debug) {
                                             log += "Using YuNet FP32 detector: " + path + "\n";
                                         }
                                         return true;
                                     } catch (const std::exception &e) {
                                         log += std::string("Failed to init YuNet FP32: ") + e.what() + "\n";
                                         return false;
                                     }
                                 }

                                 if (low == "yunet_int8") {
                                     std::string path;
                                     auto it = cfg.detector_models.find("yunet_int8");
                                     if (it != cfg.detector_models.end())
                                         path = it->second;
                                     else
                                         path = cfg.yunet_model_int8;

                                     if (path.empty() || !file_exists(path)) {
                                         log += "YuNet INT8 file not found.\n";
                                         return false;
                                     }
                                     try {
                                         det.yunet = cv::makePtr<cv::dnn::Net>(cv::dnn::readNetFromONNX(path));
                                         det.type = DetectorWrapper::DET_YUNET;
                                         det.model_path = path;

                                         int backend = parse_dnn_backend(cfg.dnn_backend);
                                         int target  = parse_dnn_target(cfg.dnn_target);
                                         det.yunet->setPreferableBackend(backend);
                                         det.yunet->setPreferableTarget(target);

                                         if (cfg.debug) {
                                             log += "Using YuNet INT8 detector: " + path + "\n";
                                         }
                                         return true;
                                     } catch (const std::exception &e) {
                                         log += std::string("Failed to init YuNet INT8: ") + e.what() + "\n";
                                         return false;
                                     }
                                 }

                                 log += "Unknown detector_profile: " + profile + "\n";
                                 return false;
                             }

                             // ==========================================================
                             // Training helpers
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

                                 std::vector<cv::String> files;
                                 cv::glob(imgdir + "/*.jpg", files, false);
                                 cv::glob(imgdir + "/*.png", files, false);

                                 if (files.empty()) {
                                     log += "No images found in: " + imgdir + "\n";
                                     return false;
                                 }

                                 std::vector<cv::Mat> faces;
                                 std::vector<int> labels;

                                 DetectorWrapper det;
                                 {
                                     FacialAuthConfig tmp = cfg;
                                     tmp.detector_profile = "haar";
                                     if (!init_detector(tmp, det, log)) {
                                         log += "Failed to initialize Haar detector for classic training.\n";
                                         return false;
                                     }
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
                                     cv::resize(gray, gray, cv::Size(92,112));

                                     faces.push_back(gray);
                                     labels.push_back(0); // un solo utente => label 0
                                 }

                                 if (faces.empty()) {
                                     log += "No valid faces found for classic training.\n";
                                     return false;
                                 }

                                 FaceRecWrapper rec;
                                 if (!rec.Create(method,
                                     cfg.lbph_threshold,
                                     cfg.eigen_threshold,
                                     cfg.fisher_threshold,
                                     cfg.eigen_components,
                                     cfg.fisher_components)) {
                                     log += "Unsupported training method: " + method + "\n";
                                 return false;
                                     }

                                     std::string err;
                                     if (!rec.Train(faces, labels, err)) {
                                         log += "Training failed (classic): " + err;
                                         return false;
                                     }

                                     if (file_exists(model_path) && !force_overwrite) {
                                         log += "Model file already exists (use --force to overwrite): " + model_path + "\n";
                                         return false;
                                     }

                                     if (!rec.Save(model_path, err)) {
                                         log += "Failed to save model: " + err;
                                         return false;
                                     }

                                     log += "Classic model saved to: " + model_path + "\n";
                                     return true;
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

                                     auto name = p.path().filename().string();

                                     if (p.path().extension() == "." + format) {
                                         try {
                                             int idx = std::stoi(p.path().stem().string());
                                             if (idx > max_idx)
                                                 max_idx = idx;
                                         }
                                         catch (...) {}
                                     }
                                 }

                                 return max_idx + 1;
                             }




                             bool fa_capture_images(const std::string &user,
                                                    const FacialAuthConfig &cfg,
                                                    const std::string &format,
                                                    std::string &log)
                             {
                                 // ---------------------------
                                 // 1. Prepara cartella utente
                                 // ---------------------------
                                 std::string img_format = format.empty()
                                 ? (cfg.image_format.empty() ? "jpg" : cfg.image_format)
                                 : format;

                                 std::string imgdir = fa_user_image_dir(cfg, user);
                                 ensure_dirs(imgdir);
                                 if (!is_dir(imgdir)) {
                                     log += "[ERRORE] Impossibile creare la directory immagini: " + imgdir + "\n";
                                     return false;
                                 }

                                 int start_index = fa_find_next_image_index(imgdir, img_format);
                                 if (cfg.debug) {
                                     std::cout << "[DEBUG] Salverò le immagini in: " << imgdir << "\n";
                                     std::cout << "[DEBUG] Prossimo indice disponibile: " << start_index << "\n";
                                 }

                                 // ---------------------------
                                 // 2. Apri webcam
                                 // ---------------------------
                                 cv::VideoCapture cap;
                                 if (!open_camera(cap, cfg, log)) {
                                     log += "[ERRORE] Impossibile aprire la webcam.\n";
                                     return false;
                                 }

                                 if (cfg.debug) {
                                     std::cout << "[DEBUG] Webcam aperta\n";
                                     std::cout << "[DEBUG] Imposto risoluzione "
                                     << cfg.width << "x" << cfg.height << "\n";
                                 }

                                 cap.set(cv::CAP_PROP_FRAME_WIDTH,  cfg.width);
                                 cap.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.height);

                                 // ---------------------------
                                 // 3. Inizializza detector
                                 // ---------------------------
                                 DetectorWrapper detector;
                                 if (!init_detector(cfg, detector, log)) {
                                     log += "[ERRORE] Impossibile inizializzare il detector (profilo=" +
                                     cfg.detector_profile + ").\n";
                                     return false;
                                 }
                                 if (cfg.debug) {
                                     std::cout << "[DEBUG] Detector inizializzato: "
                                     << (cfg.detector_profile.empty() ? "auto" : cfg.detector_profile)
                                     << "\n";
                                 }

                                 // ---------------------------
                                 // 4. Ciclo di cattura
                                 // ---------------------------
                                 int saved = 0;
                                 for (int i = 0; i < cfg.frames; ++i) {
                                     cv::Mat frame;
                                     if (!capture_frame(cap, frame, cfg, log)) {
                                         log += "[ERRORE] Frame non valido dalla webcam.\n";
                                         break;
                                     }

                                     if (cfg.verbose) {
                                         std::cout << "[VERBOSE] Frame " << (i + 1) << "/"
                                         << cfg.frames << " acquisito\n";
                                     }

                                     cv::Rect face;
                                     if (!detector.detect(frame, face)) {
                                         if (cfg.verbose) {
                                             std::cout << "[VERBOSE] Nessun volto rilevato → immagine scartata\n";
                                         }
                                         continue;
                                     }

                                     if (face.width <= 0 || face.height <= 0) {
                                         if (cfg.verbose) {
                                             std::cout << "[VERBOSE] Bounding box nulla o degenerata → immagine scartata\n";
                                         }
                                         continue;
                                     }

                                     if (cfg.debug) {
                                         std::cout << "[DEBUG] Volto rilevato: x=" << face.x
                                         << " y=" << face.y
                                         << " w=" << face.width
                                         << " h=" << face.height << "\n";
                                     }

                                     int idx = start_index + saved;
                                     std::string outfile = imgdir + "/" + std::to_string(idx) + "." + img_format;

                                     if (!cv::imwrite(outfile, frame)) {
                                         log += "[ERRORE] Impossibile salvare immagine: " + outfile + "\n";
                                     } else {
                                         ++saved;
                                         if (cfg.verbose) {
                                             std::cout << "[VERBOSE] Salvata: " << outfile << "\n";
                                         }
                                     }

                                     if (cfg.sleep_ms > 0) {
                                         sleep_ms_int(cfg.sleep_ms);
                                     }
                                 }

                                 if (saved == 0) {
                                     log += "[WARN] Nessuna immagine salvata: nessun volto rilevato nei frame acquisiti.\n";
                                     return false;
                                 }

                                 log += "[INFO] Cattura completata. Immagini salvate: " +
                                 std::to_string(saved) + "\n";
                                 return true;
                             }


                             // ==========================================================
                             // Public API: train user
                             // ==========================================================

                             bool fa_train_user(
                                 const std::string &user,
                                 const FacialAuthConfig &cfg,
                                 std::string &log
                             )
                             {
                                 std::string imgdir     = fa_user_image_dir(cfg, user);
                                 std::string model_path = fa_user_model_path(cfg, user);

                                 if (!is_dir(imgdir)) {
                                     log += "fa_train_user: image directory does not exist: " + imgdir + "\n";
                                     return false;
                                 }

                                 std::string method = cfg.training_method;
                                 std::string rp = cfg.recognizer_profile;

                                 std::string method_low = method;
                                 std::transform(method_low.begin(), method_low.end(), method_low.begin(),
                                                [](unsigned char c){ return std::tolower(c); });

                                 if (method_low == "auto") {
                                     std::string rp_low = rp;
                                     std::transform(rp_low.begin(), rp_low.end(), rp_low.begin(),
                                                    [](unsigned char c){ return std::tolower(c); });
                                     if (rp_low.rfind("sface", 0) == 0) {
                                         method = "sface";
                                     } else {
                                         method = "lbph";
                                     }
                                 }

                                 std::string mlow = method;
                                 std::transform(mlow.begin(), mlow.end(), mlow.begin(),
                                                [](unsigned char c){ return std::tolower(c); });

                                 if (mlow == "sface") {
                                     DetectorWrapper det;
                                     if (!init_detector(cfg, det, log)) {
                                         log += "fa_train_user: cannot initialize detector.\n";
                                         return false;
                                     }

                                     std::vector<cv::String> files;
                                     cv::glob(imgdir + "/*.jpg", files, false);
                                     cv::glob(imgdir + "/*.png", files, false);

                                     if (files.empty()) {
                                         log += "No images found for SFace training in: " + imgdir + "\n";
                                         return false;
                                     }

                                     std::vector<cv::Mat> embeddings;
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
                                         cv::Mat resized;
                                         cv::resize(face, resized, cv::Size(112, 112));

                                         cv::Mat emb;
                                         std::string log_emb;
                                         if (!compute_sface_embedding(cfg, resized, rp, emb, log_emb)) {
                                             log += "Failed to compute embedding for: " + fn + "\n";
                                             log += log_emb;
                                             continue;
                                         }

                                         embeddings.push_back(emb);
                                     }

                                     if (embeddings.empty()) {
                                         log += "No embeddings computed for SFace training.\n";
                                         return false;
                                     }

                                     if (file_exists(model_path) && !cfg.force_overwrite) {
                                         log += "Model file already exists (use --force to overwrite): " + model_path + "\n";
                                         return false;
                                     }

                                     if (!fa_save_sface_model(cfg, rp, model_path, embeddings)) {
                                         log += "Failed to save SFace model: " + model_path + "\n";
                                         return false;
                                     }

                                     log += "SFace model saved to: " + model_path + "\n";
                                     return true;
                                 } else {
                                     return train_classic(
                                         user,
                                         cfg,
                                         method,
                                         imgdir,
                                         model_path,
                                         cfg.force_overwrite,
                                         log
                                     );
                                 }
                             }

                             // ==========================================================
                             // Public API: test user
                             // ==========================================================

                             static double cosine_similarity(const cv::Mat &a, const cv::Mat &b)
                             {
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
                                 best_conf = 0.0;
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
                                     if (rp_low.rfind("sface", 0) == 0) {
                                         method = "sface";
                                     } else {
                                         method = "lbph";
                                     }
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

                                 if (mlow == "sface") {
                                     std::vector<cv::Mat> gallery;
                                     if (!fa_load_sface_model(modelPath, gallery)) {
                                         log += "Failed to load SFace model: " + modelPath + "\n";
                                         return false;
                                     }

                                     if (gallery.empty()) {
                                         log += "SFace model has empty gallery.\n";
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
                                         if (rp_low.find("int8") != std::string::npos) {
                                             thr = cfg.sface_int8_threshold;
                                         } else {
                                             thr = cfg.sface_fp32_threshold;
                                         }
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
                                 } else {
                                     FaceRecWrapper rec;
                                     std::string err;
                                     if (!rec.Load(modelPath, err)) {
                                         log += "Failed to load classic model: " + err;
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
                                     if (!rec.Predict(gray, label, conf, err)) {
                                         log += "Classic predict failed: " + err;
                                         return false;
                                     }

                                     best_label = label;
                                     best_conf  = conf;

                                     log += "Classic recognizer predicted label=" +
                                     std::to_string(label) + " with confidence=" +
                                     std::to_string(conf) + "\n";
                                     return true;
                                 }
                             }

                             // ==========================================================
                             // Utilities
                             // ==========================================================

                             bool fa_check_root(const char *tool_name)
                             {
                                 if (::geteuid() != 0) {
                                     std::cerr << tool_name << " must be run as root.\n";
                                     return false;
                                 }
                                 return true;
                             }

