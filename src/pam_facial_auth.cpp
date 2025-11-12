// Modulo PAM facial_auth
#include <security/pam_modules.h>
#include <security/pam_ext.h>
#include <security/pam_appl.h>
#include <syslog.h>
#include <string>
#include "FacialAuth.h"

extern "C" {

    PAM_EXTERN int pam_sm_authenticate(pam_handle_t *pamh, int flags, int argc, const char **argv) {
        pam_syslog(pamh, LOG_INFO, "pam_sm_authenticate start");

        FacialAuthConfig cfg;
        std::string trace;
        FacialAuth::load_config(cfg, argv, argc, &trace);
        if (cfg.debug) pam_syslog(pamh, LOG_DEBUG, "Config trace:\n%s", trace.c_str());

        const char *user = nullptr;
        if (pam_get_user(pamh, &user, nullptr) != PAM_SUCCESS || !user) {
            pam_syslog(pamh, LOG_ERR, "Unable to get user");
            return PAM_USER_UNKNOWN;
        }
        pam_syslog(pamh, LOG_INFO, "Authenticating user=%s", user);

        // Esegui riconoscimento
        double conf = 0.0;
        bool ok = FacialAuth::recognize_loop(cfg, user, true, pamh, conf);
        if (ok) {
            pam_syslog(pamh, LOG_INFO, "Facial auth OK (conf=%.2f)", conf);
            return PAM_SUCCESS;
        } else {
            pam_syslog(pamh, LOG_WARNING, "Facial auth FAILED");
            return PAM_AUTH_ERR;
        }
    }

    PAM_EXTERN int pam_sm_setcred(pam_handle_t *pamh, int flags, int argc, const char **argv) {
        (void)pamh; (void)flags; (void)argc; (void)argv;
        return PAM_SUCCESS;
    }

} // extern "C"
