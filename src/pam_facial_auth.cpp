#include <security/pam_modules.h>
#include <security/pam_ext.h>
#include "../include/libfacialauth.h"

PAM_EXTERN int pam_sm_authenticate(pam_handle_t *pamh, int flags, int argc, const char **argv) {
    const char *user;
    if (pam_get_user(pamh, &user, NULL) != PAM_SUCCESS) return PAM_USER_UNKNOWN;

    FacialAuthConfig cfg;
    std::string log_msg;
    fa_load_config(cfg, log_msg, FACIALAUTH_DEFAULT_CONFIG);

    std::string model_path = fa_user_model_path(cfg, user);
    if (!fa_file_exists(model_path)) return PAM_AUTHINFO_UNAVAIL;

    double confidence = 0;
    int label = -1;
    if (fa_test_user(user, cfg, model_path, confidence, label, log_msg)) {
        if (label == 0 && confidence <= cfg.lbph_threshold) return PAM_SUCCESS;
    }
    return PAM_AUTH_ERR;
}

PAM_EXTERN int pam_sm_setcred(pam_handle_t *pamh, int flags, int argc, const char **argv) {
    return PAM_SUCCESS;
}
