#include <syslog.h>

#include <security/pam_modules.h>
#include <security/pam_ext.h>
#include <security/pam_appl.h>

#include <string>
#include <cstring>

#include "../include/libfacialauth.h"

extern "C" {

    //
    // =======================================================================
    //   PAM AUTHENTICATION — nuovo comportamento con model auto-detect
    // =======================================================================
    //
    //  - Usa sempre il file di configurazione per la basedir
    //  - Deduce automaticamente il path del modello: <basedir>/models/<user>.xml
    //  - Deduce automaticamente il tipo di modello dal file XML
    //  - Nessun parametro PAM permette di cambiare modello o metodo
    //  - debug= nel pam.d resta valido
    //

    int pam_sm_authenticate(pam_handle_t *pamh, int flags, int argc, const char **argv)
    {
        const char *user = nullptr;
        const char *config_path = FACIALAUTH_CONFIG_DEFAULT;

        FacialAuthConfig cfg;
        std::string logbuf;

        bool debug_override_set = false;
        bool debug_override_val = false;

        // ---------------------------------------------------------------
        // PARSE PAM ARGUMENTS (optional)
        // ---------------------------------------------------------------
        for (int i = 0; i < argc; ++i)
        {
            if (std::strncmp(argv[i], "config=", 7) == 0)
            {
                config_path = argv[i] + 7;
            }
            else if (std::strncmp(argv[i], "debug=", 6) == 0)
            {
                debug_override_set = true;
                debug_override_val = str_to_bool(argv[i] + 6, false);
            }
        }

        // ---------------------------------------------------------------
        // GET USER
        // ---------------------------------------------------------------
        if (pam_get_user(pamh, &user, nullptr) != PAM_SUCCESS || !user || !*user)
        {
            pam_syslog(pamh, LOG_ERR, "Unable to determine username");
            return PAM_AUTH_ERR;
        }

        // ---------------------------------------------------------------
        // LOAD CONFIG
        // ---------------------------------------------------------------
        if (!read_kv_config(config_path, cfg, &logbuf))
        {
            pam_syslog(pamh, LOG_ERR, "Cannot read config file: %s", config_path);
            return PAM_AUTH_ERR;
        }

        if (debug_override_set)
            cfg.debug = debug_override_val;

        // ---------------------------------------------------------------
        // DETERMINE MODEL PATH (basedir/models/<user>.xml)
        // ---------------------------------------------------------------
        std::string model_path = fa_user_model_path(cfg, user);

        if (!file_exists(model_path))
        {
            pam_syslog(pamh, LOG_ERR,
                       "Model file for user %s not found: %s",
                       user, model_path.c_str());
            return PAM_AUTH_ERR;
        }

        // ---------------------------------------------------------------
        // DETECT MODEL TYPE FROM XML
        // ---------------------------------------------------------------
        std::string model_type = fa_detect_model_type(model_path);

        if (model_type.empty())
        {
            pam_syslog(pamh, LOG_ERR,
                       "Cannot detect model type from %s", model_path.c_str());
            return PAM_AUTH_ERR;
        }

        if (cfg.debug)
        {
            pam_syslog(pamh, LOG_DEBUG,
                       "Detected model type for %s: %s",
                       user, model_type.c_str());
        }

        // ---------------------------------------------------------------
        // START AUTHENTICATION
        // ---------------------------------------------------------------
        double best_conf = 0.0;
        int best_label = -1;

        bool ok = fa_test_user(user, cfg, model_path,
                               best_conf, best_label, logbuf);

        if (!ok)
        {
            pam_syslog(pamh, LOG_NOTICE,
                       "Face authentication FAILED for user %s (conf=%.2f threshold=%.2f)",
                       user, best_conf, cfg.threshold);
            return PAM_AUTH_ERR;
        }

        pam_syslog(pamh, LOG_INFO,
                   "Face authentication SUCCESS for user %s (conf=%.2f)",
                   user, best_conf);

        return PAM_SUCCESS;
    }

    //
    // =======================================================================
    // PAM SETCRED / ACCT_MGMT
    // =======================================================================
    //

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
