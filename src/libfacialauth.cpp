/*
 * Project: pam_facial_auth
 * License: GPL-3.0
 */

#include "libfacialauth.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <thread>
#include <regex>
#include <unistd.h>
#include <ctime>
#include <opencv2/face.hpp>

namespace fs = std::filesystem;

int get_last_index(const std::string& dir) {
    int max_idx = -1;
    if (!fs::exists(dir)) return max_idx;
    std::regex re("frame_(\\d+)\\.\\w+");
    for (const auto& entry : fs::directory_iterator(dir)) {
        std::smatch match;
        // FIX: salviamo in una variabile locale per evitare r-value temporaneo
        std::string filename = entry.path().filename().string();
        if (std::regex_match(filename, match, re)) {
            try { max_idx = std::max(max_idx, std::stoi(match[1].str())); } catch (...) {}
        }
    }
    return max_idx;
}

extern "C" {

    FA_EXPORT bool fa_check_root(const std::string& tool_name) {
        if (getuid() != 0) {
            std::cerr << "Errore: " << tool_name << " richiede privilegi di root." << std::endl;
            return false;
        }
        return true;
    }

    FA_EXPORT std::string fa_user_model_path(const FacialAuthConfig& cfg, const std::string& user) {
        return cfg.modeldir + "/" + user + ".xml";
    }

    FA_EXPORT bool fa_load_config(FacialAuthConfig& cfg, std::string& log, const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) { log = "Config mancante: " + path; return false; }
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            size_t sep = line.find('=');
            if (sep == std::string::npos) continue;
            std::string key = line.substr(0, sep), val = line.substr(sep + 1);
            key.erase(key.find_last_not_of(" \t") + 1);
            val.erase(0, val.find_first_not_of(" \t"));

            if (key == "basedir") cfg.basedir = val;
            else if (key == "modeldir") cfg.modeldir = val;
            else if (key == "device") cfg.device = val;
            else if (key == "threshold") cfg.threshold = std::stod(val);
            else if (key == "detect_yunet") cfg.detect_yunet = val;
            else if (key == "recognize_sface") cfg.recognize_sface = val;
            else if (key == "cascade_path") cfg.cascade_path = val;
            else if (key == "detector") cfg.detector = val;
            else if (key == "image_format") cfg.image_format = val;
            else if (key == "frames") cfg.frames = std::stoi(val);
            else if (key == "debug") cfg.debug = (val == "yes");
            else if (key == "verbose") cfg.verbose = (val == "yes");
            else if (key == "nogui") cfg.nogui = (val == "yes");
        }
        return true;
    }

    // Implementazione di fa_test_user (per fix errore pam_facial_auth.cpp)
    FA_EXPORT bool fa_test_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& model_path, double& confidence, int& label, std::string& log) {
        // Implementazione minima o stub per permettere la compilazione
        // Qui andrebbe la logica di caricamento del modello e predizione
        log = "Test user non ancora implementato completamente.";
        return false;
    }

    // ... (fa_capture_user e fa_train_user come nelle versioni precedenti)
}
