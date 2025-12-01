#ifndef LIBFACIALAUTH_H
#define LIBFACIALAUTH_H

#include <string>
#include <map>

//
// Path di default del file di configurazione
//
#ifndef FACIALAUTH_DEFAULT_CONFIG
#define FACIALAUTH_DEFAULT_CONFIG "/etc/security/pam_facial.conf"
#endif

struct FacialAuthConfig
{
    std::string basedir;
    std::string device;

    int width  = 640;
    int height = 480;

    int frames   = 20;
    int sleep_ms = 200;

    bool fallback_device    = false;
    bool debug              = false;
    bool verbose            = false;
    bool nogui              = false;
    bool force_overwrite    = false;
    bool ignore_failure     = false;
    bool save_failed_images = false;

    std::string detector_profile;
    std::string recognizer_profile;

    double lbph_threshold    = 80.0;
    double eigen_threshold   = 5000.0;
    double fisher_threshold  = 500.0;

    int eigen_components   = 80;
    int fisher_components  = 80;

    double sface_threshold      = 0.5;
    double sface_fp32_threshold = 0.5;
    double sface_int8_threshold = 0.5;

    std::string model_path;
    std::string haar_cascade_path;

    std::string yunet_model;
    std::string yunet_model_int8;

    std::string sface_model;
    std::string sface_model_int8;

    std::map<std::string,std::string> detector_models;
    std::map<std::string,std::string> recognizer_models;

    std::string image_format = "jpg";
};

bool fa_load_config(FacialAuthConfig&, std::string &log, const std::string &path);
bool fa_check_root(const std::string &toolname);
std::string fa_user_image_dir(const FacialAuthConfig&, const std::string &user);
std::string fa_user_model_path(const FacialAuthConfig&, const std::string &user);

bool fa_capture_images(const std::string &user,
                       const FacialAuthConfig &cfg,
                       const std::string &format,
                       std::string &log);

bool fa_train_user(const std::string &user,
                   const FacialAuthConfig &cfg,
                   std::string &log);

bool fa_test_user(const std::string &user,
                  const FacialAuthConfig &cfg,
                  const std::string &model_path,
                  double &best_conf,
                  int &best_label,
                  std::string &log,
                  double threshold_override=-1.0);

#endif
