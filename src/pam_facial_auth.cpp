/*
 * Project: pam_facial_auth
 * License: GPL-3.0
 *
 * Uses fixed model path:
 *   /etc/security/pam_facial_auth/<user>.xml
 */

#include <security/pam_modules.h>
#include <security/pam_ext.h>
#include <syslog.h>

#include "libfacialauth.h"

PAM_EXTERN int pam_sm_authenticate(pam_handle_t* pamh, int /*flags*/, int /*argc*/, const char** /*argv*/) {
    const char* user = nullptr;
    if (pam_get_user(pamh, &user, nullptr) != PAM_SUCCESS || !user) {
        return PAM_AUTH_ERR;
    }

    FacialAuthConfig cfg;
    std::string log;
    fa_load_config(cfg, log, ""); // default config path inside library

    const std::string model_path = fa_user_model_path(cfg, user);

    double confidence = 0.0;
    int label = -1;

    const bool ok = fa_test_user(user, cfg, model_path, confidence, label, log);
    if (!ok) {
        pam_syslog(pamh, LOG_ERR,
                   "pam_facial_auth: FAIL user=%s model=%s confidence=%f label=%d (%s)",
                   user, model_path.c_str(), confidence, label, log.c_str());
        return cfg.ignore_failure ? PAM_IGNORE : PAM_AUTH_ERR;
    }

    pam_syslog(pamh, LOG_INFO,
               "pam_facial_auth: OK user=%s model=%s confidence=%f label=%d (%s)",
               user, model_path.c_str(), confidence, label, log.c_str());

    return PAM_SUCCESS;
}

PAM_EXTERN int pam_sm_setcred(pam_handle_t* /*pamh*/, int /*flags*/, int /*argc*/, const char** /*argv*/) {
    return PAM_SUCCESS;
}
