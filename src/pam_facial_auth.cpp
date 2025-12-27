#include "../include/libfacialauth.h"
#include <security/pam_modules.h>
#include <security/pam_ext.h>
#include <syslog.h>

PAM_EXTERN int pam_sm_authenticate(pam_handle_t *pamh, int flags, int argc, const char **argv) {
    const char *user_ptr;
    if (pam_get_user(pamh, &user_ptr, NULL) != PAM_SUCCESS) return PAM_USER_UNKNOWN;

    FacialAuthConfig cfg; std::string log;
    fa_load_config(cfg, log, FACIALAUTH_DEFAULT_CONFIG);

    double conf; int label;
    if (fa_test_user(user_ptr, cfg, fa_user_model_path(cfg, user_ptr), conf, label, log)) {
        return PAM_SUCCESS;
    }
    return cfg.ignore_failure ? PAM_IGNORE : PAM_AUTH_ERR;
}

PAM_EXTERN int pam_sm_setcred(pam_handle_t *pamh, int flags, int argc, const char **argv) { return PAM_SUCCESS; }
