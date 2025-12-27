#ifndef LIBFACIALAUTH_H
#define LIBFACIALAUTH_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

// Struttura di configurazione allineata con i file sorgente
struct FacialAuthConfig {
    bool verbose = false;
    bool debug = false;
    bool force = false;
    int sleep_ms = 100;           // Ritardo tra i frame durante la cattura
    std::string image_format = "jpg";
    std::string model_type = "lbph";
    std::string cascade_path = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml";
    double threshold = 80.0;
};

// --- API della Libreria ---

// Funzioni di sistema
bool fa_check_root(const std::string& tool_name);
bool fa_file_exists(const std::string& path);

// Funzioni di cattura e training
bool fa_capture_dataset(const FacialAuthConfig& cfg, std::string& log, const std::string& user, int count);
bool fa_capture_user(const std::string& user, const FacialAuthConfig& cfg, const std::string& detector, std::string& log);
bool fa_train_user(const std::string& user, const FacialAuthConfig& cfg, std::string& log);

// Funzioni di verifica
bool fa_test_user(const std::string& user, const FacialAuthConfig& cfg, std::string& log);

#endif // LIBFACIALAUTH_H
