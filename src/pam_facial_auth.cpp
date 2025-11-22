#include <security/pam_modules.h>
#include <security/pam_ext.h>
#include <security/pam_appl.h>
#include <syslog.h>
#include <cstring>
#include <string>

#include "../include/libfacialauth.h"

extern "C" {

    // =======================================================================
    // Helper: get user
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
    // Helper: parse pam arguments (only debug=1 or debug=true)
    // =======================================================================

    static void parse_pam_args(int argc, const char **argv, FacialAuthConfig &cfg, pam_handle_t *pamh)
    {
        for (int i = 0; i < argc; i++) {
            if (!argv[i])
                continue;

            std::string arg = argv[i];

            // Format: debug=1 OR debug=true
            if (arg.rfind("debug=", 0) == 0) {
                std::string val = arg.substr(6);

                if (val == "1" || val == "true" || val == "True" || val == "TRUE") {
                    cfg.debug = true;
                    pam_syslog(pamh, LOG_DEBUG, "PAM argument: debug FORCED to true");
                }
            }
        }
    }

    // =======================================================================
    // PAM AUTHENTICATE
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

        // Load configuration file
        fa_load_config(FACIALAUTH_CONFIG_DEFAULT, cfg, log);

        // Apply PAM runtime arguments (debug=1)
        parse_pam_args(argc, argv, cfg, pamh);

        // Determine threshold based on the model XML
        double thr_used = cfg.threshold;
        std::string model_path = fa_user_model_path(cfg, user);

        if (file_exists(model_path)) {
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
                                   "Model for %s: DNN enabled, threshold=%.4f",
                                   user.c_str(), thr_used);
                    }
                }
            }
        }

        double best_conf = 0.0;
        int best_label   = -1;

        bool ok = fa_test_user(user, cfg, model_path, best_conf, best_label, log);

        if (cfg.debug) {
            pam_syslog(pamh, LOG_DEBUG,
                       "FaceAuth user=%s conf=%.6f label=%d thr=%.6f",
                       user.c_str(), best_conf, best_label, thr_used);
        }

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
    // Other PAM functions
    // =======================================================================

    PAM_EXTERN int pam_sm_setcred(pam_handle_t *pamh, int flags,
