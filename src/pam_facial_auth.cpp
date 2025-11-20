#include "../include/libfacialauth.h"
#include <security/pam_modules.h>
#include <security/pam_ext.h>
#include <syslog.h>        // <<< NECESSARIO per LOG_ERR, LOG_INFO, LOG_DEBUG, ...
#include <cstring>
#include <string>

// ==========================================================
//  PAM module: pam_facial_auth
// ==========================================================
//
//  - Config path default: /etc/security/pam_facial.conf
//  - CLI arguments in PAM stack override config file
//  - Debug attivo solo se debug=true nel config oppure stack PAM
// ==========================================================

extern "C" {

    int pam_sm_authenticate(pam_handle_t *pamh, int flags, int argc, const char **argv) {

        const char *user = nullptr;
        const char *config_path = "/etc/security/pam_facial.conf";

        bool debug_override = false;

        // ----------------------------------------------------------
        // Parse PAM arguments
        // ----------------------------------------------------------
        for (int i = 0; i < argc; i++) {
            if (strncmp(argv[i], "config=", 7) == 0) {
                config_path = argv[i] + 7;
            }
            else if (strncmp(argv[i], "debug=", 6) == 0) {
                const char *val = argv[i] + 6;
                debug_override =
                (strcmp(val, "true") == 0) ||
                (strcmp(val, "1") == 0) ||
                (strcasecmp(val, "yes") == 0);
            }
        }

        // ----------------------------------------------------------
        // Get the user
        // ----------------------------------------------------------
        if (pam_get_user(pamh, &user, nullptr) != PAM_SUCCESS || !user) {
            pam_syslog(pamh, LOG_ERR, "Unable to obtain username");
            return PAM_AUTH_ERR;
        }

        // ----------------------------------------------------------
        // Load config
        // ----------------------------------------------------------
        FacialAuthConfig cfg;
        std::string logbuf;

        if (!read_kv_config(config_path, cfg, &logbuf)) {
            pam_syslog(pamh, LOG_WARNING,
                       "Cannot read config file: %s", config_path);
        }

        // Debug override from PAM stack
        if (debug_override)
            cfg.debug = true;

        // ----------------------------------------------------------
        // Prepare model path
        // ----------------------------------------------------------
        std::string model = fa_user_model_path(cfg, user);

        double best_confidence = 9999.0;
        int best_label = -1;

        // ----------------------------------------------------------
        // Perform face authentication
        // ----------------------------------------------------------
        bool ok = fa_test_user(user, cfg, model,
                               best_confidence, best_label, logbuf);

        if (cfg.debug) {
            pam_syslog(pamh, LOG_DEBUG,
                       "FaceAuth user=%s conf=%.2f label=%d",
                       user, best_confidence, best_label);
        }

        if (!ok) {
            pam_syslog(pamh, LOG_NOTICE,
                       "Face authentication FAILED for user %s", user);
            return PAM_AUTH_ERR;
        }

        pam_syslog(pamh, LOG_INFO,
                   "Face authentication SUCCESSFUL for user %s", user);

        return PAM_SUCCESS;
    }

    int pam_sm_setcred(pam_handle_t *pamh, int flags, int argc, const char **argv) {
        return PAM_SUCCESS;
    }

} // extern "C"
