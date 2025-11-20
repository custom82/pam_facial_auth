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
    //  - Usa libfacialauth::fa_test_user per fare l’autenticazione
    //  - Non gestiamo per ora parametri PAM extra (optional, debug, ecc.)
    // =======================================================================

    static int get_pam_user(pam_handle_t *pamh, std::string &user)
    {
        const char *puser = nullptr;
        int pam_err = pam_get_user(pamh, &puser, "login: ");
        if (pam_err != PAM_SUCCESS || !puser || !*puser) {
            pam_syslog(pamh, LOG_ERR, "Unable to get PAM user");
            return PAM_USER_UNKNOWN;
        }
        user = puser;
        return PAM_SUCCESS;
    }

    PAM_EXTERN int pam_sm_authenticate(pam_handle_t *pamh,
                                       int flags,
                                       int argc,
                                       const char **argv)
    {
        (void)flags;
        (void)argc;
        (void)argv;

        std::string user;
        int ret = get_pam_user(pamh, user);
        if (ret != PAM_SUCCESS)
            return ret;

        FacialAuthConfig cfg;
        std::string log;

        // Carica config predefinita
        fa_load_config(FACIALAUTH_CONFIG_DEFAULT, cfg, log);

        double best_conf = 0.0;
        int best_label   = -1;

        bool ok = fa_test_user(user, cfg, "", best_conf, best_label, log);

        if (cfg.debug) {
            pam_syslog(pamh, LOG_DEBUG,
                       "FaceAuth user=%s conf=%.4f label=%d",
                       user.c_str(), best_conf, best_label);
        }

        if (!ok) {
            pam_syslog(pamh, LOG_NOTICE,
                       "Face authentication FAILED for user %s (conf=%.4f thr=%.4f)",
                       user.c_str(), best_conf, cfg.threshold);
            return PAM_AUTH_ERR;
        }

        pam_syslog(pamh, LOG_INFO,
                   "Face authentication SUCCESS for user %s (conf=%.4f thr=%.4f)",
                   user.c_str(), best_conf, cfg.threshold);
        return PAM_SUCCESS;
    }

    PAM_EXTERN int pam_sm_setcred(pam_handle_t *pamh, int flags,
                                  int argc, const char **argv)
    {
        (void)pamh;
        (void)flags;
        (void)argc;
        (void)argv;
        return PAM_SUCCESS;
    }

    PAM_EXTERN int pam_sm_open_session(pam_handle_t *pamh, int flags,
                                       int argc, const char **argv)
    {
        (void)pamh;
        (void)flags;
        (void)argc;
        (void)argv;
        return PAM_SUCCESS;
    }

    PAM_EXTERN int pam_sm_close_session(pam_handle_t *pamh, int flags,
                                        int argc, const char **argv)
    {
        (void)pamh;
        (void)flags;
        (void)argc;
        (void)argv;
        return PAM_SUCCESS;
    }

    PAM_EXTERN int pam_sm_acct_mgmt(pam_handle_t *pamh, int flags,
                                    int argc, const char **argv)
    {
        (void)pamh;
        (void)flags;
        (void)argc;
        (void)argv;
        return PAM_SUCCESS;
    }

} // extern "C"
