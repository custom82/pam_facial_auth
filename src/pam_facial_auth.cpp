#include <security/pam_modules.h>
#include <security/pam_ext.h>
#include <security/pam_appl.h>
#include <syslog.h>

#include <cstring>
#include <string>

#include "../include/libfacialauth.h"

extern "C" {

    /*
     * =====================================================================
     *  PAM facial authentication
     * =====================================================================
     *
     *  - Carica configurazione da pam_facial.conf
     *  - Determina automaticamente il modello da:
     *        <basedir>/models/<user>.xml
     *  - Usa le soglie specifiche nel file di configurazione:
     *        lbph_threshold=
     *        eigen_threshold=
     *        fisher_threshold=
     *  - Non accetta parametri particolari dal PAM (solo config= debug=)
     *  - In caso di errore -> ritorna sempre PAM_AUTH_ERR
     *
     */

    int pam_sm_authenticate(pam_handle_t *pamh, int flags,
                            int argc, const char **argv)
    {
        (void)flags;

        const char *user = nullptr;
        const char *config_path = FACIALAUTH_CONFIG_DEFAULT;

        FacialAuthConfig cfg;
        std::string logbuf;

        bool debug_override_set = false;
        bool debug_override_val = false;

        // ---------------------------------------------------------
        // Parse argomenti da stack PAM
        // ---------------------------------------------------------
        for (int i = 0; i < argc; i++) {
            if (strncmp(argv[i], "config=", 7) == 0) {
                config_path = argv[i] + 7;
            } else if (strncmp(argv[i], "debug=", 6) == 0) {
                debug_override_set = true;
                debug_override_val = str_to_bool(argv[i] + 6, false);
            }
        }

        // ---------------------------------------------------------
        // Ottieni l’utente
        // ---------------------------------------------------------
        if (pam_get_user(pamh, &user, nullptr) != PAM_SUCCESS ||
            !user || !*user)
        {
            pam_syslog(pamh, LOG_ERR, "Unable to determine username");
            return PAM_AUTH_ERR;
        }

        // ---------------------------------------------------------
        // Carica configurazione
        // ---------------------------------------------------------
        if (!read_kv_config(config_path, cfg, &logbuf)) {
            pam_syslog(pamh, LOG_ERR,
                       "Cannot read config file: %s",
                       config_path);
            return PAM_AUTH_ERR;
        }

        if (debug_override_set)
            cfg.debug = debug_override_val;

        // ---------------------------------------------------------
        // Determina il modello utente:
        //     <basedir>/models/<user>.xml
        // ---------------------------------------------------------
        std::string model_path = fa_user_model_path(cfg, user);

        if (!file_exists(model_path)) {
            pam_syslog(pamh, LOG_ERR,
                       "Model file not found for user %s: %s",
                       user, model_path.c_str());
            return PAM_AUTH_ERR;
        }

        // ---------------------------------------------------------
        // Esegue la verifica
        // ---------------------------------------------------------
        double best_conf = 0.0;
        int best_label = -1;

        bool ok = fa_test_user(
            user,
            cfg,
            model_path,
            best_conf,
            best_label,
            logbuf,
            -1.0        // NO override threshold — usa quella del modello
        );

        if (!ok) {
            pam_syslog(pamh, LOG_NOTICE,
                       "FaceAuth FAILED for %s (conf=%.2f)",
                       user, best_conf);
            return PAM_AUTH_ERR;
        }

        pam_syslog(pamh, LOG_INFO,
                   "FaceAuth SUCCESS for %s (conf=%.2f)",
                   user, best_conf);

        return PAM_SUCCESS;
    }


    // =====================================================================
    // NO-OP
    // =====================================================================

    int pam_sm_setcred(pam_handle_t *pamh, int flags,
                       int argc, const char **argv)
    {
        (void)pamh; (void)flags; (void)argc; (void)argv;
        return PAM_SUCCESS;
    }

    int pam_sm_acct_mgmt(pam_handle_t *pamh, int flags,
                         int argc, const char **argv)
    {
        (void)pamh; (void)flags; (void)argc; (void)argv;
        return PAM_SUCCESS;
    }

} // extern "C"
