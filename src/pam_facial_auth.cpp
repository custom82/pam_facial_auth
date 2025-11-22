#include <security/pam_modules.h>
#include <security/pam_ext.h>
#include <security/pam_appl.h>
#include <syslog.h>

#include <string>
#include <filesystem>
#include <opencv2/core.hpp>

#include "../include/libfacialauth.h"

namespace fs = std::filesystem;

extern "C" {

    // =======================================================================
    // Get user
    // =======================================================================

    static int get_pam_user(pam_handle_t *pamh, std::string &user)
    {
        const char *puser = nullptr;
        int pam_err = pam_get_user(pamh, &puser, "login: ");
        if (pam_err != PAM_SUCCESS || !puser || !*puser) {
            pam_syslog(pamh, LOG_ERR, "Unable to get PAM user");
            return PAM_USER_UNKNOWN;
        }
        user = puser;
        return PAM_SUCCESS;
    }

    // =======================================================================
    // Parse PAM args (only debug=1)
    // =======================================================================

    static void parse_pam_args(int argc, const char **argv, FacialAuthConfig &cfg, pam_handle_t *pamh)
    {
        for (int i = 0; i < argc; i++) {
            if (!argv[i]) continue;

            std::string arg = argv[i];

            if (arg.rfind("debug=", 0) == 0) {
                std::string val = arg.substr(6);
                if (val == "1" || val == "true" || val == "TRUE") {
                    cfg.debug = true;
                    pam_syslog(pamh, LOG_DEBUG, "PAM argument: debug forced ON");
                }
            }
        }
    }

    // =======================================================================
    // AUTHENTICATE
    // =======================================================================

    PAM_EXTERN int pam_sm_authenticate(pam_handle_t *pamh,
                                       int flags,
                                       int argc,
                                       const char **argv)
    {
        (void)flags;

        std::string user;
        int ret = get_pam_user(pamh, user);
        if (ret != PAM_SUCCESS)
            return ret;

        FacialAuthConfig cfg;
        std::string log;

        // Load default config
        fa_load_config(FACIALAUTH_CONFIG_DEFAULT, cfg, log);

        // Apply PAM runtime flags
        parse_pam_args(argc, argv, cfg, pamh);

        // Determine model path
        std::string model_path = fa_user_model_path(cfg, user);

        // Load threshold from model XML if DNN is enabled
        double thr_used = cfg.threshold;

        if (fs::exists(model_path)) {
            cv::FileStorage fs(model_path, cv::FileStorage::READ);
            if (fs.isOpened()) {
                int dnn_enabled = 0;
                fs["fa_dnn_enabled"] >> dnn_enabled;

                if (dnn_enabled == 1) {
                    double thr_model = 0.0;
                    fs["fa_dnn_threshold"] >> thr_model;

                    if (thr_model > 0.0)
                        thr_used = thr_model;

                    if (cfg.debug) {
                        pam_syslog(pamh, LOG_DEBUG,
                                   "Model %s: DNN enabled, threshold=%.4f",
                                   model_path.c_str(), thr_used);
                    }
                }
            }
        }

        double best_conf = 0.0;
        int best_label   = -1;

        bool ok = fa_test_user(user, cfg, model_path, best_conf, best_label, log);

        if (!ok) {
            pam_syslog(pamh, LOG_NOTICE,
                       "Face authentication FAILED for user %s (conf=%.6f thr=%.6f)",
                       user.c_str(), best_conf, thr_used);
            return PAM_AUTH_ERR;
        }

        pam_syslog(pamh, LOG_INFO,
                   "Face authentication SUCCESS for user %s (conf=%.6f thr=%.6f)",
                   user.c_str(), best_conf, thr_used);

        return PAM_SUCCESS;
    }

    // =======================================================================
    // Remaining PAM functions
    // =======================================================================

    PAM_EXTERN int pam_sm_setcred(pam_handle_t *pamh, int flags,
                                  int argc, const char **argv)
    {
        (void)pamh; (void)flags; (void)argc; (void)argv;
        return PAM_SUCCESS;
    }

    PAM_EXTERN int pam_sm_open_session(pam_handle_t *pamh, int flags,
                                       int argc, const char **argv)
    {
        (void)pamh; (void)flags; (void)argc; (void)argv;
        return PAM_SUCCESS;
    }

    PAM_EXTERN int pam_sm_close_session(pam_handle_t *pamh, int flags,
                                        int argc, const char **argv)
    {
        (void)pamh; (void)flags; (void)argc; (void)argv;
        return PAM_SUCCESS;
    }

    PAM_EXTERN int pam_sm_acct_mgmt(pam_handle_t *pamh, int flags,
                                    int argc, const char **argv)
    {
        (void)pamh; (void)flags; (void)argc; (void)argv;
        return PAM_SUCCESS;
    }

} // extern "C"
