#include <security/pam_appl.h>
#include <security/pam_modules.h>
#include <security/pam_ext.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include "libfacialauth.h"
#include "facial_config.h"

extern "C" {

    /* ---------------------------------------------------------
     * PAM_SM_AUTHENTICATE
     * --------------------------------------------------------- */
    PAM_EXTERN int pam_sm_authenticate(pam_handle_t *pamh, int flags,
                                       int argc, const char **argv)
    {
        const char *user = nullptr;
        pam_get_user(pamh, &user, nullptr);
        if (!user)
            return PAM_AUTH_ERR;

        /* Load configuration */
        FacialAuthConfig cfg;
        std::string cfg_err;
        if (!fa_load_config(cfg, cfg_err)) {
            pam_syslog(pamh, LOG_ERR, "pam_facial_auth: config load error: %s",
                       cfg_err.c_str());
            return PAM_AUTH_ERR;
        }

        /* Parse arguments: only 'debug' supported */
        bool force_debug = false;
        for (int i = 0; i < argc; i++) {
            if (std::string(argv[i]) == "debug") {
                force_debug = true;
            }
        }
        if (force_debug)
            cfg.debug = true;

        /* Debug info */
        if (cfg.debug) {
            pam_syslog(pamh, LOG_DEBUG, "pam_facial_auth: user=%s", user);
            pam_syslog(pamh, LOG_DEBUG, "pam_facial_auth: device=%s", cfg.device.c_str());
            pam_syslog(pamh, LOG_DEBUG, "pam_facial_auth: backend=%s target=%s",
                       cfg.dnn_backend.c_str(),
                       cfg.dnn_target.c_str());
        }

        /* Run authentication */
        std::string err;
        bool result = fa_authenticate_user(cfg, user, err);

        if (!result) {
            if (cfg.debug)
                pam_syslog(pamh, LOG_DEBUG, "pam_facial_auth: FAIL: %s", err.c_str());
            return PAM_AUTH_ERR;
        }

        if (cfg.debug)
            pam_syslog(pamh, LOG_DEBUG, "pam_facial_auth: SUCCESS for user=%s", user);

        return PAM_SUCCESS;
    }

    /* ---------------------------------------------------------
     * PAM_SM_SETCRED
     * --------------------------------------------------------- */
    PAM_EXTERN int pam_sm_setcred(pam_handle_t *pamh, int flags,
                                  int argc, const char **argv)
    {
        return PAM_SUCCESS;
    }

} // extern "C"
