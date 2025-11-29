#include "../include/libfacialauth.h"

#include <security/pam_appl.h>
#include <security/pam_modules.h>
#include <security/pam_ext.h>

#include <syslog.h>
#include <sys/stat.h>
#include <unistd.h>

#include <string>
#include <cstring>
#include <cstdarg>

// ==========================================================
// Logging PAM helper
// ==========================================================

static void pam_log(pam_handle_t *pamh, int priority, const char *fmt, ...)
{
    char buf[512];

    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);

    pam_syslog(pamh, priority, "%s", buf);
}

// ==========================================================
// PAM AUTH MODULE
// ==========================================================

extern "C"
int pam_sm_authenticate(pam_handle_t *pamh, int flags,
                        int argc, const char **argv)
{
    (void)flags;
    (void)argc;
    (void)argv;

    // -----------------------------
    // 1) ottieni username
    // -----------------------------
    const char *user_c = nullptr;
    int ret = pam_get_user(pamh, &user_c, "Username: ");

    if (ret != PAM_SUCCESS || !user_c || !*user_c) {
        pam_log(pamh, LOG_ERR,
                "pam_facial_auth: cannot obtain user name");
        return PAM_AUTH_ERR;
    }

    std::string user(user_c);

    // -----------------------------
    // 2) carica configurazione
    // -----------------------------
    FacialAuthConfig cfg;
    std::string logbuf;

    if (!fa_load_config(cfg, logbuf, FACIALAUTH_CONFIG_DEFAULT)) {
        pam_log(pamh, LOG_ERR,
                "pam_facial_auth: cannot load config %s",
                FACIALAUTH_CONFIG_DEFAULT);

        if (cfg.ignore_failure)
            return PAM_IGNORE;

        return PAM_AUTH_ERR;
    }

    if (!logbuf.empty())
        pam_log(pamh, LOG_INFO, "cfg load: %s", logbuf.c_str());

    // -----------------------------
    // 3) determina tipo riconoscitore
    // -----------------------------
    std::string rp = cfg.recognizer_profile;
    for (char &c : rp)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    bool use_sface = (rp == "sface" || rp == "sface_int8");

    // -----------------------------
    // 4) modello classico solo se serve
    // -----------------------------
    std::string model_path;

    if (!use_sface) {
        // modello LBPH/Eigen/Fisher: path XML
        model_path = fa_user_model_path(cfg, user);
    } else {
        // SFace: embeddings da file (se esiste) oppure immagini
        model_path.clear();
    }

    // -----------------------------
    // 5) test autenticazione
    // -----------------------------
    double      best_conf  = 0.0;
    int         best_label = -1;
    std::string err;

    bool ok = fa_test_user(
        user,            // utente
        cfg,             // configurazione
        model_path,      // modello (vuoto per SFace)
    best_conf,       // out: conf / coseno
    best_label,      // out
    err,             // logbuf (errori)
    -1.0             // override soglia (none)
    );

    // -----------------------------
    // 6) risultato
    // -----------------------------
    if (!ok) {
        pam_log(pamh, LOG_NOTICE,
                "pam_facial_auth: auth failed for user '%s' (%s)",
                user.c_str(),
                err.c_str());

        if (cfg.ignore_failure)
            return PAM_IGNORE;

        return PAM_AUTH_ERR;
    }

    pam_log(pamh, LOG_INFO,
            "pam_facial_auth: auth OK for '%s' (best=%.3f)",
            user.c_str(),
            best_conf);

    return PAM_SUCCESS;
}

// ==========================================================
// PAM SETCRED (no-op)
// ==========================================================

extern "C"
int pam_sm_setcred(pam_handle_t *pamh, int flags,
                   int argc, const char **argv)
{
    (void)pamh;
    (void)flags;
    (void)argc;
    (void)argv;

    return PAM_SUCCESS;
}
