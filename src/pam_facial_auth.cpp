#include <security/pam_appl.h>
#include <security/pam_misc.h>
#include <security/pam_modules.h>
#include <security/pam_ext.h>  // Aggiunto pam_ext.h
#include <opencv2/opencv.hpp>
#include "libfacialauth.h"  // Includi il file header per la classe FacialAuth
#include <syslog.h>

int pam_sm_authenticate(pam_handle_t *pamh, int flags, int argc, const char **argv) {
    const char *user = NULL;
    int retval;

    // Ottieni l'utente da PAM
    retval = pam_get_user(pamh, &user, NULL);
    if (retval != PAM_SUCCESS) {
        pam_syslog(pamh, LOG_ERR, "Impossibile ottenere l'utente");
        return retval;
    }

    // Crea un'istanza della classe FacialAuth per autenticazione
    FacialAuth facialAuth;

    // Esegui l'autenticazione facciale
    if (!facialAuth.Authenticate(user)) {
        pam_syslog(pamh, LOG_ERR, "Autenticazione facciale fallita per l'utente: %s", user);
        return PAM_AUTH_ERR;
    }

    pam_syslog(pamh, LOG_INFO, "Autenticazione facciale riuscita per l'utente: %s", user);
    return PAM_SUCCESS;
}

