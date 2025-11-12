#include <security/pam_appl.h>
#include <security/pam_misc.h>
#include <security/pam_modules.h>
#include <opencv2/opencv.hpp>
#include "FacialAuth.h"  // Include the FacialAuth header for authentication class

int pam_sm_authenticate(pam_handle_t *pamh, int flags, int argc, const char **argv) {
    const char *user = NULL;
    int retval;

    // Retrieve the user
    retval = pam_get_user(pamh, &user, NULL);
    if (retval != PAM_SUCCESS) {
        pam_syslog(pamh, LOG_ERR, "Unable to obtain user");
        return retval;
    }

    // Instantiate the FacialAuth class (Ensure this class is defined in "FacialAuth.h")
    FacialAuth facialAuth;

    // Authenticate the user using facial recognition
    if (!facialAuth.Authenticate(user)) {
        pam_syslog(pamh, LOG_ERR, "Autenticazione facciale fallita");
        return PAM_AUTH_ERR;
    }

    pam_syslog(pamh, LOG_INFO, "Autenticazione facciale riuscita");
    return PAM_SUCCESS;
}
