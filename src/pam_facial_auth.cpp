#include "../include/libfacialauth.h"

#include <security/pam_appl.h>
#include <security/pam_modules.h>
#include <security/pam_ext.h>

#include <string>
#include <string.h>
#include <syslog.h>

// ---------------------------------------------------------------------
// HELPERS
// ---------------------------------------------------------------------

static void log_pam(pam_handle_t *pamh, int level, const char *fmt, ...)
{
    char buf[1024];

    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);

    pam_syslog(pamh, level, "%s", buf);
}

// ---------------------------------------------------------------------
// pam_sm_authenticate — REQUIRED
// ---------------------------------------------------------------------

PAM_EXTERN int pam_sm_authenticate(pam_handle_t *pamh, int flags,
                                   int argc, const char **argv)
{
    const char *user = nullptr;

    // -------------------------------------------------------------
    // Recupera username
    // -------------------------------------------------------------
    if (pam_get_user(pamh, &user, "Username: ") != PAM_SUCCESS || !user) {
        log_pam(pamh, LOG_ERR, "pam_facial_auth: cannot obtain username");
        return PAM_AUTH_ERR;
    }

    // -------------------------------------------------------------
    // Carica configurazione
    // -------------------------------------------------------------
    FacialAuthConfig cfg;
    std::string logbuf;

    if (!fa_load_config(cfg, logbuf, FACIALAUTH_CONFIG_DEFAULT)) {
        log_pam(pamh, LOG_ERR,
                "pam_facial_auth: cannot load config '%s'",
                FACIALAUTH_CONFIG_DEFAULT);
        return PAM_AUTH_ERR;
    }

    if (!logbuf.empty())
        log_pam(pamh, LOG_INFO, "%s", logbuf.c_str());
    logbuf.clear();

    // -------------------------------------------------------------
    // Esegui test facciale (SFace o Classic)
    // -------------------------------------------------------------
    double best_conf  = 0.0;
    int    best_label = -1;

    bool ok = fa_test_user(
        user,
        cfg,
        cfg.model_path,     // modello automatico (basato su basedir)
    best_conf,
    best_label,
    logbuf,
    -1.0                // soglia override disabilitata
    );

    if (!logbuf.empty())
        log_pam(pamh, LOG_INFO, "%s", logbuf.c_str());

    // -------------------------------------------------------------
    // Risultato
    // -------------------------------------------------------------
    if (ok) {
        log_pam(pamh, LOG_INFO,
                "pam_facial_auth: AUTH SUCCESS for user '%s' (conf=%.3f)",
                user, best_conf);
        return PAM_SUCCESS;
    }

    log_pam(pamh, LOG_ERR,
            "pam_facial_auth: AUTH FAILED for user '%s'",
            user);

    return PAM_AUTH_ERR;
}


// ---------------------------------------------------------------------
// pam_sm_setcred — generalmente noop
// ---------------------------------------------------------------------
PAM_EXTERN int pam_sm_setcred(pam_handle_t *pamh, int flags,
                              int argc, const char **argv)
{
    (void)pamh; (void)flags; (void)argc; (void)argv;
    return PAM_SUCCESS;
}


// ---------------------------------------------------------------------
// pam_sm_open_session — opzionale (noop)
// ---------------------------------------------------------------------
PAM_EXTERN int pam_sm_open_session(pam_handle_t *pamh, int flags,
                                   int argc, const char **argv)
{
    (void)pamh; (void)flags; (void)argc; (void)argv;
    return PAM_SUCCESS;
}


// ---------------------------------------------------------------------
// pam_sm_close_session — opzionale (noop)
// ---------------------------------------------------------------------
PAM_EXTERN int pam_sm_close_session(pam_handle_t *pamh, int flags,
                                    int argc, const char **argv)
{
    (void)pamh; (void)flags; (void)argc; (void)argv;
    return PAM_SUCCESS;
}

