#include <security/pam_modules.h>
#include <security/pam_ext.h>
#include <security/pam_appl.h>
#include <syslog.h>
#include <string>

#include "../include/libfacialauth.h"

extern "C" {

    // =======================================================================
    // PAM AUTHENTICATION
    // =======================================================================

    int pam_sm_authenticate(pam_handle_t *pamh, int flags, int argc, const char **argv)
    {
        const char *user = nullptr;
        const char *config_path = "/etc/security/pam_facial.conf";   // <── DEFAULT FISSO

        FacialAuthConfig cfg;
        std::string logbuf;

        // -------------------------------------------------------------------
        // Parse MODULE ARGUMENTS
        // -------------------------------------------------------------------
        for (int i = 0; i < argc; i++) {
            if (strncmp(argv[i], "config=", 7) == 0) {
                config_path = argv[i] + 7;
            } else if (strncmp(argv[i], "debug=", 6) == 0) {
                cfg.debug = str_to_bool(argv[i] + 6, false);
            }
        }

        // -------------------------------------------------------------------
        // Get username
        // -------------------------------------------------------------------
        if (pam_get_user(pamh, &user, nullptr) != PAM_SUCCESS || !user) {
            pam_syslog(pamh, LOG_ERR, "Unable to obtain username");
            return PAM_AUTH_ERR;
        }

        // -------------------------------------------------------------------
        // Load configuration file
        // -------------------------------------------------------------------
        if (!read_kv_config(config_path, cfg, &logbuf)) {
            pam_syslog(pamh, LOG_WARNING, "Cannot read config file: %s", config_path);
            return PAM_AUTH_ERR;
        }

        // -------------------------------------------------------------------
        // Face Authentication
        // -------------------------------------------------------------------
        double best_conf = 9999;
        int best_label = -1;
        std::string model_path = fa_user_model_path(cfg, user);

        bool ok = fa_test_user(user, cfg, model_path, best_conf, best_label, logbuf);

        // Log only if debug=true
        if (cfg.debug) {
            pam_syslog(pamh, LOG_DEBUG, "FaceAuth user=%s conf=%.2f label=%d",
                       user, best_conf, best_label);
        }

        if (!ok) {
            pam_syslog(pamh, LOG_NOTICE, "Face authentication FAILED for user %s", user);
            return PAM_AUTH_ERR;
        }

        pam_syslog(pamh, LOG_INFO, "Face authentication SUCCESSFUL for user %s", user);
        return PAM_SUCCESS;
    }

    // =======================================================================
    // PAM ACCOUNT
    // =======================================================================

    int pam_sm_acct_mgmt(pam_handle_t *pamh, int flags, int argc, const char **argv)
    {
        return PAM_SUCCESS;
    }

} // extern "C"
