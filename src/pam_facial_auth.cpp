#include "../include/libfacialauth.h"
#include <security/pam_modules.h>
#include <security/pam_ext.h>
#include <syslog.h>

PAM_EXTERN int pam_sm_authenticate(pam_handle_t *pamh, int flags, int argc, const char **argv) {
    const char* user = nullptr;
    if (pam_get_user(pamh, &user, NULL) != PAM_SUCCESS || !user) return PAM_USER_UNKNOWN;

    FacialAuthConfig cfg; std::string log;
    if (!fa_load_config(cfg, log, FACIALAUTH_DEFAULT_CONFIG)) return PAM_AUTHINFO_UNAVAIL;

    std::string m_path = fa_user_model_path(cfg, user);
    if (!fa_file_exists(m_path)) return PAM_AUTHINFO_UNAVAIL;

    double conf = 0.0; int lbl = -1;
    cfg.nogui = true;

    if (fa_test_user(user, cfg, m_path, conf, lbl, log)) {
        bool ok = (cfg.training_method == "sface") ? (conf >= cfg.sface_threshold) : (conf <= cfg.lbph_threshold);
        if (ok) return PAM_SUCCESS;
    }
    return PAM_AUTH_ERR;
}

PAM_EXTERN int pam_sm_setcred(pam_handle_t *pamh, int flags, int argc, const char **argv) { return PAM_SUCCESS; }
