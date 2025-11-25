#include <security/pam_modules.h>
#include <security/pam_ext.h>
#include <security/pam_appl.h>
#include <syslog.h>
#include <cstring>
#include <string>

#include "../include/libfacialauth.h"

extern "C" {

    // =======================================================================
    // PAM AUTHENTICATION — wrapper minimalista
    // =======================================================================
    //
    // Tutta la logica di:
    //   – lettura config
    //   – parsing parametri
    //   – gestione camera
    //   – caricamento modello
    //   – face detection & recognition
    //
    // È IMPLENTATA IN libfacialauth.so
    //
    // Questo file chiama solo: fa_test_user()
    // =======================================================================

    int pam_sm_authenticate(pam_handle_t *pamh, int flags, int argc, const char **argv)
    {
        const char *user = nullptr;

        // percorso config di default
        const char *config_path = FACIALAUTH_CONFIG_DEFAULT;

        FacialAuthConfig cfg;     // default già impostati in libfacialauth.h
        std::string logbuf;

        bool debug_override_set = false;
        bool debug_override_val = false;

        // -------------------------------------------------------------------
        // Parametri passati dal blocco PAM
        // -------------------------------------------------------------------
        for (int i = 0; i < argc; i++) {
            if (std::strncmp(argv[i], "config=", 7) == 0) {
                config_path = argv[i] + 7;

            } else if (std::strncmp(argv[i], "debug=", 6) == 0) {
                debug_override_set = true;
                debug_override_val = str_to_bool(argv[i] + 6, false);
            }
        }

        // -------------------------------------------------------------------
        // Ottieni utente
        // -------------------------------------------------------------------
        if (pam_get_user(pamh, &user, nullptr) != PAM_SUCCESS || !user || !*user) {
            pam_syslog(pamh, LOG_ERR, "Unable to obtain username");
            return PAM_AUTH_ERR;
        }

        // -------------------------------------------------------------------
        // Carica configurazione
        // -------------------------------------------------------------------
        if (!read_kv_config(config_path, cfg, &logbuf)) {
            pam_syslog(pamh, LOG_ERR, "Cannot read config file: %s", config_path);
            return PAM_AUTH_ERR;
        }

        // Applica override da parametri PAM
        if (debug_override_set) {
            cfg.debug = debug_override_val;
        }

        // -------------------------------------------------------------------
        // Percorso del modello per l’utente
        // -------------------------------------------------------------------
        std::string model_path = fa_user_model_path(cfg, user);

        // Preferenze usate da fa_test_user
        double best_conf  = 9999.0;
        int    best_label = -1;

        // -------------------------------------------------------------------
        // Autenticazione biometrica
        // -------------------------------------------------------------------
        bool ok = fa_test_user(user, cfg, model_path, best_conf, best_label, logbuf);

        // Log dettagliato solo in debug
        if (cfg.debug) {
            pam_syslog(pamh, LOG_DEBUG,
                       "FaceAuth: user=%s conf=%.2f label=%d",
                       user, best_conf, best_label);
        }

        if (!ok) {
            // Fallimento: bloccare sempre
            pam_syslog(pamh, LOG_NOTICE,
                       "Face authentication FAILED for user %s", user);
            return PAM_AUTH_ERR;
        }

        if (cfg.debug) {
            pam_syslog(pamh, LOG_INFO,
                       "Face authentication SUCCESS for user %s", user);
        }

        return PAM_SUCCESS;
    }

    // =======================================================================
    // PAM SETCRED / ACCT_MGMT — no-op
    // =======================================================================

    int pam_sm_setcred(pam_handle_t *pamh, int flags, int argc, const char **argv)
    {
        (void)pamh; (void)flags; (void)argc; (void)argv;
        return PAM_SUCCESS;
    }

    int pam_sm_acct_mgmt(pam_handle_t *pamh, int flags, int argc, const char **argv)
    {
        (void)pamh; (void)flags; (void)argc; (void)argv;
        return PAM_SUCCESS;
    }

} // extern "C"
