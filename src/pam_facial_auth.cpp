#include "../include/libfacialauth.h"
#include <security/pam_modules.h>
#include <security/pam_ext.h>
#include <syslog.h>
#include <string>
#include <vector>
#include <cstring>

extern "C" {

    PAM_EXTERN int pam_sm_authenticate(
        pam_handle_t *pamh,
        int /*flags*/,
        int argc,
        const char **argv)
    {
        const char *user = nullptr;
        bool cli_debug = false;

        // -------------------------------------------------------
        // Leggi parametro debug da pam.d
        // -------------------------------------------------------
        for (int i = 0; i < argc; ++i) {
            if (strcmp(argv[i], "debug") == 0) {
                cli_debug = true;
            }
        }

        // -------------------------------------------------------
        // Ottieni user PAM
        // -------------------------------------------------------
        if (pam_get_user(pamh, &user, nullptr) != PAM_SUCCESS || !user) {
            pam_syslog(pamh, LOG_ERR, "Cannot get PAM user");
            return PAM_AUTH_ERR;
        }

        // -------------------------------------------------------
        // Carica configurazione
        // -------------------------------------------------------
        FacialAuthConfig cfg;
        std::string cfg_err;

        if (!fa_load_config(cfg, cfg_err, FACIALAUTH_CONFIG_DEFAULT)) {
            pam_syslog(pamh, LOG_ERR, "Config load failed: %s", cfg_err.c_str());
            return PAM_AUTH_ERR;
        }

        // CLI debug ha precedenza sul file di configurazione
        if (cli_debug)
            cfg.debug = true;

        // -------------------------------------------------------
        // APRI il syslog con facility AUTH
        // -------------------------------------------------------
        openlog("pam_facial_auth", LOG_PID, LOG_AUTH);

        // -------------------------------------------------------
        // Wrapper logging con pam_syslog
        // -------------------------------------------------------
        auto pam_log = [&](const char *fmt, ...) {
            int level = cfg.debug ? LOG_DEBUG : LOG_INFO;

            va_list args;
            va_start(args, fmt);
            pam_vsyslog(pamh, level, fmt, args);
            va_end(args);
        };

        // -------------------------------------------------------
        // Log iniziale
        // -------------------------------------------------------
        pam_log("Starting facial authentication for user %s", user);

        if (cfg.debug)
            pam_log("Debug mode enabled (cfg.debug=1)");

        // -------------------------------------------------------
        // Esegue il test del volto
        // -------------------------------------------------------
        double best_conf = 0.0;
        int best_label   = -1;
        std::string test_log;

        bool ok = fa_test_user(
            user,
            cfg,
            cfg.model_path,
            best_conf,
            best_label,
            test_log,
            -1
        );

        // -------------------------------------------------------
        // Log dei dettagli (info o debug)
        // -------------------------------------------------------
        if (!test_log.empty()) {
            pam_log("%s", test_log.c_str());
        }

        if (ok)
            pam_log("Authentication OK for user %s", user);
        else
            pam_log("Authentication FAILED for user %s", user);

        // -------------------------------------------------------
        // Chiudi syslog
        // -------------------------------------------------------
        closelog();

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
