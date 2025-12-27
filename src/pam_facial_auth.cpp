#include "../include/libfacialauth.h"
#include <security/pam_modules.h>
#include <security/pam_ext.h>
#include <syslog.h>

PAM_EXTERN int pam_sm_authenticate(pam_handle_t *pamh, int flags, int argc, const char **argv) {
    const char* user;
    if (pam_get_user(pamh, &user, NULL) != PAM_SUCCESS || !user) return PAM_USER_UNKNOWN;

    FacialAuthConfig cfg;
    std::string log;
    if (!fa_load_config(cfg, log, FACIALAUTH_DEFAULT_CONFIG)) {
        pam_syslog(pamh, LOG_ERR, "Could not load config: %s", log.c_str());
        return PAM_AUTHINFO_UNAVAIL;
    }

    std::string model_path = fa_user_model_path(cfg, user);
    if (!fa_file_exists(model_path)) return PAM_AUTH_ERR;

    double confidence = 0.0;
    int label = -1;

    // Forza nogui a true per il modulo PAM (non vogliamo finestre pop-up durante il login)
    cfg.nogui = true;

    if (fa_test_user(user, cfg, model_path, confidence, label, log)) {
        bool success = false;

        // Logica Threshold basata sul metodo
        if (cfg.training_method == "sface") {
            // SFace usa la Cosine Similarity (più alto è meglio, tipicamente > 0.36)
            success = (confidence >= cfg.sface_threshold);
        } else {
            // Metodi classici (LBPH) usano la distanza (più basso è meglio, tipicamente < 60-80)
            success = (confidence <= cfg.lbph_threshold);
        }

        if (success) {
            pam_syslog(pamh, LOG_INFO, "Facial Auth successful for user %s (score: %f)", user, confidence);
            return PAM_SUCCESS;
        }
    }

    pam_syslog(pamh, LOG_NOTICE, "Facial Auth failed for user %s (score: %f)", user, confidence);
    return PAM_AUTH_ERR;
}

PAM_EXTERN int pam_sm_setcred(pam_handle_t *pamh, int flags, int argc, const char **argv) {
    return PAM_SUCCESS;
}
