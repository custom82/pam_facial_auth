#include "../include/libfacialauth.h"
#include <security/pam_modules.h>
#include <security/pam_ext.h>
#include <syslog.h>

/*
 * pam_sm_authenticate: Il cuore del modulo.
 * Gestisce il riconoscimento e decide il ritorno in base ai threshold.
 */
PAM_EXTERN int pam_sm_authenticate(pam_handle_t *pamh, int flags, int argc, const char **argv) {
    const char* user = nullptr;
    int retval;

    // 1. Ottieni lo username
    retval = pam_get_user(pamh, &user, NULL);
    if (retval != PAM_SUCCESS || !user) {
        return PAM_USER_UNKNOWN;
    }

    // 2. Carica configurazione
    FacialAuthConfig cfg;
    std::string log_msg;
    if (!fa_load_config(cfg, log_msg, FACIALAUTH_DEFAULT_CONFIG)) {
        pam_syslog(pamh, LOG_ERR, "Configuration error for user %s: %s", user, log_msg.c_str());
        return PAM_AUTHINFO_UNAVAIL;
    }

    // 3. Verifica esistenza modello
    std::string model_path = fa_user_model_path(cfg, user);
    if (!fa_file_exists(model_path)) {
        if (cfg.debug) pam_syslog(pamh, LOG_DEBUG, "Model missing for user %s at %s", user, model_path.c_str());
        return PAM_AUTH_ERR;
    }

    // 4. Esecuzione Test Face Recognition
    double confidence = 0.0;
    int label = -1;
    cfg.nogui = true; // Obbligatorio in ambiente PAM

    if (fa_test_user(user, cfg, model_path, confidence, label, log_msg)) {
        bool authenticated = false;

        // Logica soglie (Thresholding)
        if (cfg.training_method == "sface") {
            authenticated = (confidence >= cfg.sface_threshold);
        } else {
            authenticated = (confidence <= cfg.lbph_threshold);
        }

        if (authenticated) {
            pam_syslog(pamh, LOG_INFO, "Facial Auth SUCCESS: user %s (score: %f)", user, confidence);
            return PAM_SUCCESS;
        } else {
            pam_syslog(pamh, LOG_NOTICE, "Facial Auth REJECTED: user %s (score: %f, threshold: %f)",
                       user, confidence, (cfg.training_method == "sface" ? cfg.sface_threshold : cfg.lbph_threshold));
        }
    } else {
        pam_syslog(pamh, LOG_ERR, "Facial Auth ERROR for user %s: %s", user, log_msg.c_str());
    }

    return PAM_AUTH_ERR;
}

PAM_EXTERN int pam_sm_setcred(pam_handle_t *pamh, int flags, int argc, const char **argv) {
    return PAM_SUCCESS;
}
