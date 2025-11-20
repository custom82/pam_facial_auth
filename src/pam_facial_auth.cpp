#include <security/pam_modules.h>
#include <security/pam_ext.h>
#include <security/pam_appl.h>
#include <syslog.h>
#include <cstring>
#include <string>

#include "../include/libfacialauth.h"

extern "C" {

    // =======================================================================
    // PAM AUTHENTICATION
    // =======================================================================
    //
    //  - Config predefinito: /etc/security/pam_facial.conf
    //  - Argomenti PAM (config=, debug=) hanno precedenza sul file
    //  - Senza debug: modulo silenzioso (niente output a terminale)
    //  - Con debug: log solo su syslog (pam_syslog), mai su stdout/stderr
    //  - In caso di fallimento: sempre PAM_AUTH_ERR
    // =======================================================================

    int pam_sm_authenticate(pam_handle_t *pamh, int flags, int argc, const char **argv)
    {
        const char *user = nullptr;

        // default fisso del config
        const char *config_path = "/etc/security/pam_facial.conf";

        FacialAuthConfig cfg;      // con i default definiti in libfacialauth.h
        std::string logbuf;

        // Per fare sì che debug= passato via PAM abbia precedenza sul file
        bool debug_override_set = false;
        bool debug_override_val = false;

        // -------------------------------------------------------------------
        // Parse MODULE ARGUMENTS (stack PAM)
        // -------------------------------------------------------------------
        for (int i = 0; i < argc; i++) {
            if (std::strncmp(argv[i], "config=", 7) == 0) {
                config_path = argv[i] + 7;
            } else if (std::strncmp(argv[i], "debug=", 6) == 0) {
                debug_override_set = true;
                debug_override_val = str_to_bool(argv[i] + 6, false);
            }
            // altri argomenti vengono ignorati dal modulo PAM
        }

        // -------------------------------------------------------------------
        // Ottiene l'utente
        // -------------------------------------------------------------------
        if (pam_get_user(pamh, &user, nullptr) != PAM_SUCCESS || !user || !*user) {
            // errore serio: lo logghiamo sempre
            pam_syslog(pamh, LOG_ERR, "Unable to obtain username");
            return PAM_AUTH_ERR;
        }

        // -------------------------------------------------------------------
        // Carica configurazione da file
        // -------------------------------------------------------------------
        if (!read_kv_config(config_path, cfg, &logbuf)) {
            // impossibile leggere la configurazione → fallisce sempre
            pam_syslog(pamh, LOG_ERR, "Cannot read config file: %s", config_path);
            return PAM_AUTH_ERR;
        }

        // debug= nello stack PAM ha precedenza sul file
        if (debug_override_set) {
            cfg.debug = debug_override_val;
        }

        // -------------------------------------------------------------------
        // Prepara il percorso del modello utente
        // -------------------------------------------------------------------
        std::string model_path = fa_user_model_path(cfg, user);

        double best_conf  = 9999.0;
        int    best_label = -1;

        // -------------------------------------------------------------------
        // Esegue la verifica biometrica
        // -------------------------------------------------------------------
        bool ok = fa_test_user(user, cfg, model_path, best_conf, best_label, logbuf);

        // Log di dettaglio solo se debug attivo
        if (cfg.debug) {
            pam_syslog(pamh, LOG_DEBUG,
                       "FaceAuth user=%s conf=%.2f label=%d",
                       user, best_conf, best_label);
        }

        if (!ok) {
            // Fallimento autenticazione: blocca sempre (comportamento A)
            pam_syslog(pamh, LOG_NOTICE,
                       "Face authentication FAILED for user %s", user);
            return PAM_AUTH_ERR;
        }

        // Successo: messaggio info solo se debug attivo (per mantenere quiet di default)
        if (cfg.debug) {
            pam_syslog(pamh, LOG_INFO,
                       "Face authentication SUCCESSFUL for user %s", user);
        }

        return PAM_SUCCESS;
    }

    // =======================================================================
    // PAM SETCRED / ACCT_MGMT: moduli "no-op", sempre success
    // =======================================================================

    int pam_sm_setcred(pam_handle_t *pamh, int flags, int argc, const char **argv)
    {
        (void)pamh;
        (void)flags;
        (void)argc;
        (void)argv;
        return PAM_SUCCESS;
    }

    int pam_sm_acct_mgmt(pam_handle_t *pamh, int flags, int argc, const char **argv)
    {
        (void)pamh;
        (void)flags;
        (void)argc;
        (void)argv;
        return PAM_SUCCESS;
    }

} // extern "C"
