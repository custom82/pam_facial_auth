#include "../include/libfacialauth.h"

#include <security/pam_modules.h>
#include <security/pam_ext.h>

#include <string>
#include <vector>
#include <syslog.h>

extern "C" {

    PAM_EXTERN int pam_sm_authenticate(
        pam_handle_t *pamh,
        int /*flags*/,
        int /*argc*/,
        const char **/*argv*/)
    {
        const char *user = nullptr;

        if (pam_get_user(pamh, &user, nullptr) != PAM_SUCCESS || !user) {
            pam_syslog(pamh, LOG_ERR, "Cannot get PAM user");
            return PAM_AUTH_ERR;
        }

        // ============================================
        // Load config
        // ============================================
        FacialAuthConfig cfg;
        std::string cfg_err;

        if (!fa_load_config(cfg, cfg_err, FACIALAUTH_CONFIG_DEFAULT)) {
            pam_syslog(pamh, LOG_ERR,
                       "Config load failed: %s", cfg_err.c_str());
            return PAM_AUTH_ERR;
        }

        // ============================================
        // Run authentication using unified API
        // ============================================
        double best_conf = 0.0;
        int best_label   = -1;
        std::string test_log;

        bool ok = fa_test_user(
            user,
            cfg,
            cfg.model_path,    // se vuoto, usa modello utente
            best_conf,
            best_label,
            test_log,
            -1                 // threshold override
        );

        if (!test_log.empty()) {
            pam_syslog(pamh,
                       ok ? LOG_INFO : LOG_ERR,
                       "%s", test_log.c_str());
        }

        return ok ? PAM_SUCCESS : PAM_AUTH_ERR;
    }

    PAM_EXTERN int pam_sm_setcred(
        pam_handle_t * /*pamh*/,
        int /*flags*/,
        int /*argc*/,
        const char ** /*argv*/)
    {
        return PAM_SUCCESS;
    }

} // extern "C"
