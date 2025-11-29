#include <security/pam_appl.h>
#include <security/pam_modules.h>
#include <security/pam_ext.h>

// -------------------------------------------------------------
// pam_sm_authenticate — versione REQUIRED indipendente da pam_unix
// -------------------------------------------------------------

PAM_EXTERN int pam_sm_authenticate(pam_handle_t *pamh, int flags,
                                   int argc, const char **argv)
{
    const char *user = nullptr;

    // Ottieni username
    if (pam_get_user(pamh, &user, "Username: ") != PAM_SUCCESS || !user) {
        pam_syslog(pamh, LOG_ERR, "pam_facial_auth: cannot obtain username");
        return PAM_AUTH_ERR;  // blocca (required)
    }

    // Carica configurazione
    FacialAuthConfig cfg;
    std::string logbuf;
    if (!fa_load_config(cfg, logbuf, FACIALAUTH_CONFIG_DEFAULT)) {
        pam_syslog(pamh, LOG_ERR,
                   "pam_facial_auth: cannot load config: %s",
                   FACIALAUTH_CONFIG_DEFAULT);
        return PAM_AUTH_ERR;
    }

    if (!logbuf.empty())
        pam_syslog(pamh, LOG_INFO, "%s", logbuf.c_str());
    logbuf.clear();

    // Esegui autenticazione facciale
    double best_conf  = 0.0;
    int    best_label = -1;

    bool ok = fa_test_user(
        user,
        cfg,
        cfg.model_path,   // modello automatico da basedir/models/user.xml
        best_conf,
        best_label,
        logbuf,
        -1.0              // niente soglia override → usa quella del config
    );

    if (!logbuf.empty())
        pam_syslog(pamh, LOG_INFO, "%s", logbuf.c_str());

    if (ok) {
        pam_syslog(pamh, LOG_INFO,
                   "pam_facial_auth: AUTH SUCCESS for user '%s' (conf=%.3f)",
                   user, best_conf
        );
        return PAM_SUCCESS;
    }

    pam_syslog(pamh, LOG_ERR,
               "pam_facial_auth: AUTH FAILED for user '%s'", user);

    return PAM_AUTH_ERR;
}
