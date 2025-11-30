#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include "libfacialauth.h"   // tua libreria

using namespace std;

/* ============================================================
 *  PARSING ARGOMENTI CLI
 * ============================================================ */
static void print_help() {
    cout << "Usage: facial_test -u <user> [options]\n"
    << "Options:\n"
    << "  -u, --user <name>           Utente da testare\n"
    << "  -d, --device <path>         Dispositivo video (default /dev/video0)\n"
    << "      --backend <cpu|cuda>    Backend DNN\n"
    << "      --target <cpu|cuda>     Target DNN\n"
    << "      --detector <auto|haar|yunet>\n"
    << "  --debug                     Abilita debug\n"
    << "  -h, --help                  Mostra questo help\n";
}

/* ============================================================
 *  MAIN LOGICA DI TEST
 * ============================================================ */
static bool run_test(const FacialAuthConfig& cfg,
                     const string& user,
                     const string& device,
                     bool debug)
{
    if (debug) {
        cout << "[INFO] Testing SFace model for user " << user
        << " on " << device << "\n";
    }

    cv::VideoCapture cap(device);
    if (!cap.isOpened()) {
        cerr << "[ERROR] Cannot open video device " << device << "\n";
        return false;
    }

    cv::Mat frame;
    cap >> frame;

    if (frame.empty()) {
        cerr << "[ERROR] Empty frame from device\n";
        return false;
    }

    // ============================================
    //  Caricamento embedding utente
    // ============================================
    vector<float> embeddings;
    if (!fa_load_user_embeddings(cfg, user, embeddings)) {
        cerr << "[ERROR] No SFace gallery features for user\n";
        return false;
    }

    // ============================================
    //  Caricamento modello SFace (con backend/target/cuda)
    // ============================================
    cv::dnn::Net net;
    string err;

    if (!load_sface_model_dnn(cfg, cfg.sface_model, net, err)) {
        cerr << "[ERROR] Failed loading SFace model: " << err << "\n";
        return false;
    }

    // ============================================
    //  Selettore del detector
    // ============================================
    Detector det;

    if (!fa_init_detector(det, cfg, err)) {
        if (debug)
            cerr << "[DEBUG] Detector '" << cfg.detector_profile
            << "' failed, fallback to Haar\n";

        cfg.detector_profile = "haar";
        if (!fa_init_detector(det, cfg, err)) {
            cerr << "[ERROR] Cannot initialize any detector\n";
            return false;
        }
    }

    if (debug)
        cout << "[DEBUG] Detector selected: " << det.name << "\n";

    // ============================================
    //  Estrarre volto
    // ============================================
    cv::Rect faceBox;
    if (!det.detect(frame, faceBox)) {
        cerr << "[ERROR] No face detected\n";
        return false;
    }

    cv::Mat face = frame(faceBox).clone();

    // ============================================
    //  Calcolo feature con SFace
    // ============================================
    vector<float> feat;
    if (!fa_extract_sface(net, face, feat, err)) {
        cerr << "[ERROR] Failed extracting SFace features\n";
        return false;
    }

    // ============================================
    //  Similarità con embedding utente
    // ============================================
    float sim = fa_cosine_similarity(embeddings, feat);

    cout << "[INFO] SFace similarity = " << sim
    << " (threshold " << cfg.sface_threshold << ")\n";

    bool ok = sim >= cfg.sface_threshold;

    // ============================================
    //  RISULTATO FINALE (HUMAN FRIENDLY)
    // ============================================
    if (ok) cout << "[RESULT] AUTH OK\n";
    else    cout << "[RESULT] AUTH FAILED\n";

    return ok;
}

/* ============================================================
 *  MAIN CLI
 * ============================================================ */
int main(int argc, char** argv)
{
    string user;
    string device = "/dev/video0";
    string backend_cli = "";
    string target_cli  = "";
    string detector_cli = "";
    bool debug = false;

    // --- parse CLI ---
    for (int i = 1; i < argc; i++) {
        string a = argv[i];

        if (a == "-u" || a == "--user") {
            if (i + 1 < argc) user = argv[++i];
        }
        else if (a == "-d" || a == "--device") {
            if (i + 1 < argc) device = argv[++i];
        }
        else if (a == "--backend") {
            if (i + 1 < argc) backend_cli = argv[++i];
        }
        else if (a == "--target") {
            if (i + 1 < argc) target_cli = argv[++i];
        }
        else if (a == "--detector") {
            if (i + 1 < argc) detector_cli = argv[++i];
        }
        else if (a == "--debug") {
            debug = true;
        }
        else if (a == "-h" || a == "--help") {
            print_help();
            return 0;
        }
    }

    if (user.empty()) {
        cerr << "[ERROR] Missing -u <user>\n";
        return 1;
    }

    // Carica configurazione
    FacialAuthConfig cfg;
    string err;
    if (!fa_load_config(cfg, err, FACIALAUTH_CONFIG_DEFAULT)) {
        cerr << "[ERROR] Cannot load config: " << err << "\n";
        return 1;
    }

    // Override CLI
    if (!backend_cli.empty()) cfg.dnn_backend = backend_cli;
    if (!target_cli.empty())  cfg.dnn_target  = target_cli;
    if (!detector_cli.empty()) cfg.detector_profile = detector_cli;

    return run_test(cfg, user, device, debug) ? 0 : 1;
}
