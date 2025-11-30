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
        // 1. Controllo parametro debug passato dal modulo PAM
        // -------------------------------------------------------
        for (int i = 0; i < argc; ++i) {
            if (strcmp(argv[i], "debug") == 0) {
                cli_debug = true;
            }
        }

        // -------------------------------------------------------
        // 2. Ottieni user PAM
        // -------------------------------------------------------
        if (pam_get_user(pamh, &user, nullptr) != PAM_SUCCESS || !user) {
            pam_syslog(pamh, LOG_ERR, "Cannot get PAM user");
            return PAM_AUTH_ERR;
        }

        // -------------------------------------------------------
        // 3. Carica configurazione
        // -------------------------------------------------------
        FacialAuthConfig cfg;
        std::string cfg_err;

        if (!fa_load_config(cfg, cfg_err, FACIALAUTH_CONFIG_DEFAULT)) {
            pam_syslog(pamh, LOG_ERR, "Config load failed: %s", cfg_err.c_str());
            return PAM_AUTH_ERR;
        }

        // CLI ha precedenza sul file di configurazione
        if (cli_debug)
            cfg.debug = true;

        // -------------------------------------------------------
        // 4. Logging di debug (con facility AUTH)
        // -------------------------------------------------------
        if (cfg.debug) {
            pam_syslog(pamh, LOG_DEBUG, "Debug mode enabled for user %s", user);
        }

        // -------------------------------------------------------
        // 5. Esegue il test del volto
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
        // 6. Log dettagliati solo se debug attivo
        // -------------------------------------------------------
        if (cfg.debug && !test_log.empty()) {
            pam_syslog(pamh,
                       ok ? LOG_DEBUG : LOG_ERR,
                       "%s", test_log.c_str());
        }

        // -------------------------------------------------------
        // 7. Ritorna il risultato a PAM
        // -------------------------------------------------------
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
