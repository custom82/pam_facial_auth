#include <security/pam_modules.h>
#include <security/pam_ext.h>
#include <security/pam_appl.h>

#include <string>
#include <cstring>

#include "../include/libfacialauth.h"

extern "C" {

    // pam_sm_authenticate - funzione principale di autenticazione
    PAM_EXTERN int pam_sm_authenticate(pam_handle_t *pamh, int flags,
                                       int argc, const char **argv)
    {
        (void)flags;
        FacialAuthConfig cfg; // con default

        std::string config_path = "/etc/pam_facial_auth/pam_facial.conf";

        // parametri PAM aggiuntivi (es: config=/path, threshold=...)
        for (int i = 0; i < argc; ++i) {
            std::string a = argv[i];
            auto pos = a.find('=');
            if (pos == std::string::npos) continue;
            std::string key = a.substr(0, pos);
            std::string val = a.substr(pos + 1);
            if (key == "config") {
                config_path = val;
            } else if (key == "threshold") {
                cfg.threshold = std::stod(val);
            } else if (key == "device") {
                cfg.device = val;
            } else if (key == "nogui") {
                cfg.nogui = str_to_bool(val, cfg.nogui);
            } else if (key == "debug") {
                cfg.debug = str_to_bool(val, cfg.debug);
            }
        }

        std::string log;
        read_kv_config(config_path, cfg, &log);

        const char *user_c = nullptr;
        int pret = pam_get_user(pamh, &user_c, "Username: ");
        if (pret != PAM_SUCCESS || !user_c || !*user_c) {
            pam_syslog(pamh, LOG_ERR, "Unable to get PAM user");
            return PAM_AUTH_ERR;
        }
        std::string user(user_c);

        // Usiamo il percorso modello di default basedir/models/<user>.xml
        std::string model_path = fa_user_model_path(cfg, user);

        double best_conf;
        int best_label;
        bool ok = fa_test_user(user, cfg, model_path, best_conf, best_label, log);

        if (cfg.debug) {
            pam_syslog(pamh, LOG_DEBUG, "pam_facial_auth log:\n%s", log.c_str());
        }

        if (!ok) {
            pam_info(pamh, "Facial authentication failed (conf=%.2f, thr=%.2f)",
                     best_conf, cfg.threshold);
            return PAM_AUTH_ERR;
        }

        pam_info(pamh, "Facial authentication succeeded (conf=%.2f <= %.2f)",
                 best_conf, cfg.threshold);
        return PAM_SUCCESS;
    }

    // pam_sm_setcred - qui non facciamo nulla
    PAM_EXTERN int pam_sm_setcred(pam_handle_t *pamh, int flags,
                                  int argc, const char **argv)
    {
        (void)pamh; (void)flags; (void)argc; (void)argv;
        return PAM_SUCCESS;
    }

} // extern "C"
