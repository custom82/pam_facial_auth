/*
 * Project: pam_facial_auth
 * License: GPL-3.0
 */

#include <security/pam_modules.h>
#include <security/pam_ext.h>
#include "libfacialauth.h"

PAM_EXTERN int pam_sm_authenticate(pam_handle_t *pamh, int flags, int argc, const char **argv) {
    const char* user;
    if (pam_get_user(pamh, &user, NULL) != PAM_SUCCESS || !user) return PAM_AUTH_ERR;

    FacialAuthConfig cfg;
    std::string log;
    fa_load_config(cfg, log);

    double confidence = 0.0;
    int label = -1;
    std::string model_path = fa_user_model_path(cfg, user);

    if (!fa_test_user(user, cfg, model_path, confidence, label, log)) {
        pam_syslog(pamh, LOG_ERR, "Facial Auth fallito: %s", log.c_str());
        return PAM_AUTH_ERR;
    }

    // Usiamo il 'threshold' generico definito nella struct
    if (confidence <= cfg.threshold) {
        return PAM_SUCCESS;
    }

    return PAM_AUTH_ERR;
}

PAM_EXTERN int pam_sm_setcred(pam_handle_t *pamh, int flags, int argc, const char **argv) {
    return PAM_SUCCESS;
}
