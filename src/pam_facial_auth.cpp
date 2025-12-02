#include "../include/libfacialauth.h"

extern "C" {
    #include <security/pam_modules.h>
    #include <security/pam_ext.h>
    #include <syslog.h>
    #include <security/pam_appl.h>
}

#include <string>

static const char *DEFAULT_CONFIG_PATH = "/etc/security/pam_facial.conf";

static int get_pam_user(pam_handle_t *pamh, std::string &user)
{
    const char *puser = nullptr;
    int ret = pam_get_user(pamh, &puser, nullptr);
    if (ret != PAM_SUCCESS || !puser || !*puser)
        return ret;
    user = puser;
    return PAM_SUCCESS;
}

extern "C" {

    PAM_EXTERN int pam_sm_authenticate(
        pam_handle_t *pamh,
        int flags,
        int argc,
        const char **argv
    )
    {
        (void)flags;

        std::string cfg_path = DEFAULT_CONFIG_PATH;
        bool ignore_failure_override = false;

        for (int i = 0; i < argc; ++i) {
            std::string opt = argv[i] ? argv[i] : "";
            if (opt.rfind("config=", 0) == 0) {
                cfg_path = opt.substr(7);
            } else if (opt == "ignore_failure") {
                ignore_failure_override = true;
            }
        }

        std::string user;
        int pret = get_pam_user(pamh, user);
        if (pret != PAM_SUCCESS) {
            pam_syslog(pamh, LOG_ERR, "pam_facial_auth: cannot get user");
            return pret;
        }

        FacialAuthConfig cfg;
        std::string log;
        if (!fa_load_config(cfg, log, cfg_path)) {
            pam_syslog(pamh, LOG_ERR, "pam_facial_auth: failed to load config: %s", log.c_str());
            // Se la config non c'è, meglio fallire "pulito" o ignorare?
            return PAM_AUTH_ERR;
        }

        if (ignore_failure_override)
            cfg.ignore_failure = true;

        std::string model_path = fa_user_model_path(cfg, user);
        double best_conf = 0.0;
        int best_label = -1;
        bool ok = fa_test_user(user, cfg, model_path, best_conf, best_label, log, -1.0);

        pam_syslog(pamh, LOG_INFO, "pam_facial_auth: %s", log.c_str());

        if (ok) {
            return PAM_SUCCESS;
        } else {
            if (cfg.ignore_failure) {
                return PAM_IGNORE;
            }
            return PAM_AUTH_ERR;
        }
    }

    PAM_EXTERN int pam_sm_setcred(
        pam_handle_t *pamh,
        int flags,
        int argc,
        const char **argv
    )
    {
        (void)pamh;
        (void)flags;
        (void)argc;
        (void)argv;
        return PAM_SUCCESS;
    }

} // extern "C"
