#include <security/pam_modules.h>
#include <security/pam_ext.h>
#include "FacialAuth.h"
#include <opencv2/opencv.hpp>

extern "C" {

    PAM_EXTERN int pam_sm_authenticate(pam_handle_t *pamh, int flags, int argc, const char **argv) {
        const char* user;
        pam_get_user(pamh, &user, NULL);

        FacialAuth facialAuth;

        if (!facialAuth.Authenticate(user)) {
            pam_syslog(pamh, LOG_ERR, "Autenticazione facciale fallita");
            return PAM_AUTH_ERR;
        }

        pam_syslog(pamh, LOG_INFO, "Autenticazione facciale riuscita");
        return PAM_SUCCESS;
    }

    PAM_EXTERN int pam_sm_setcred(pam_handle_t *pamh, int flags, int argc, const char **argv) {
        return PAM_SUCCESS;
    }

}
