#include "../include/libfacialauth.h"
#include <security/pam_modules.h>
#include <security/pam_ext.h>
#include <syslog.h>

/* Funzione principale chiamata da PAM durante l'autenticazione */
PAM_EXTERN int pam_sm_authenticate(pam_handle_t *pamh, int flags, int argc, const char **argv) {
    const char *user;
    int retval = pam_get_user(pamh, &user, NULL);
    if (retval != PAM_SUCCESS || !user) return PAM_USER_UNKNOWN;

    FacialAuthConfig cfg;
    std::string log_msg;

    // Carichiamo la configurazione
    fa_load_config(cfg, log_msg, FACIALAUTH_DEFAULT_CONFIG);

    // Percorso del modello dell'utente
    std::string model_path = fa_user_model_path(cfg, user);

    // Se il modello non esiste, passiamo il controllo al modulo successivo (es. password)
    if (!fa_file_exists(model_path)) {
        return PAM_AUTHINFO_UNAVAIL;
    }

    double confidence = 0.0;
    int label = -1;

    // Eseguiamo il test (non interattivo)
    // Usiamo fa_test_user direttamente per evitare output su stdout (proibito in PAM)
    bool authenticated = fa_test_user(user, cfg, model_path, confidence, label, log_msg);

    if (authenticated) {
        pam_syslog(pamh, LOG_INFO, "Facial auth success for user %s (conf: %f)", user, confidence);
        return PAM_SUCCESS;
    }

    pam_syslog(pamh, LOG_NOTICE, "Facial auth failed for user %s: %s", user, log_msg.c_str());

    // Se fallisce, restituiamo errore di autenticazione per procedere con gli altri moduli
    return PAM_AUTH_ERR;
}

/* Altre funzioni PAM obbligatorie (stub) */
PAM_EXTERN int pam_sm_setcred(pam_handle_t *pamh, int flags, int argc, const char **argv) {
    return PAM_SUCCESS;
}

PAM_EXTERN int pam_sm_acct_mgmt(pam_handle_t *pamh, int flags, int argc, const char **argv) {
    return PAM_SUCCESS;
}
