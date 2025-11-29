#include <security/pam_modules.h>
#include <security/pam_ext.h>
#include <security/pam_appl.h>

#include <string>
#include <sys/stat.h>
#include <syslog.h>

#include "libfacialauth.h"

using std::string;


/* ---------------------------------------------------------
 * Usa file_exists del header (NON ridefinirlo)
 * --------------------------------------------------------- */


/* ---------------------------------------------------------
 *  PAM ENTRY POINT
 * --------------------------------------------------------- */
extern "C" int pam_sm_authenticate(
    pam_handle_t *pamh,
    int flags,
    int argc,
    const char **argv)
{
    const char *user_c = nullptr;
    pam_get_user(pamh, &user_c, "Username: ");

    if (!user_c)
        return PAM_AUTH_ERR;

    string user = user_c;

    // -----------------------------------------------------
    // 1) Carica configurazione
    // -----------------------------------------------------
    string cfg_path = FACIALAUTH_CONFIG_DEFAULT;
    FacialAuthConfig cfg;
    string err;

    if (!fa_read_config(cfg_path, cfg, err)) {
        pam_syslog(pamh, LOG_ERR,
                   "pam_facial_auth: config error: %s",
                   err.c_str());
        return PAM_AUTH_ERR;
    }

    // -----------------------------------------------------
    // 2) Path modello utente
    //    <basedir>/models/<user>.xml
    // -----------------------------------------------------
    string model_path = cfg.basedir + "/models/" + user + ".xml";

    if (!file_exists(model_path)) {
        pam_syslog(pamh, LOG_ERR,
                   "pam_facial_auth: model not found: %s",
                   model_path.c_str());
        return PAM_AUTH_ERR;
    }

    // -----------------------------------------------------
    // 3) Esegui autenticazione
    // -----------------------------------------------------
    double best_conf = 0.0;
    int method = 0;
    string err2;

    bool ok = fa_test_user(
        user,         // username
        cfg,          // configurazione
        model_path,   // path XML
        best_conf,    // best confidence
        method,       // metodo usato
        err2,         // messaggio errore
        0.0           // override threshold
    );

    if (!ok) {
        pam_syslog(pamh, LOG_ERR,
                   "pam_facial_auth: authentication failed: %s",
                   err2.c_str());
        return PAM_AUTH_ERR;
    }

    pam_syslog(pamh, LOG_INFO,
               "pam_facial_auth: user %s authenticated (method=%d, conf=%f)",
               user.c_str(), method, best_conf);

    return PAM_SUCCESS;
}


/* ---------------------------------------------------------
 *  PAM SETCRED (non usato)
 * --------------------------------------------------------- */
extern "C" int pam_sm_setcred(
    pam_handle_t *pamh, int flags, int argc, const char **argv)
{
    return PAM_SUCCESS;
}
