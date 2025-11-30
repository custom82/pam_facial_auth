#include "../include/libfacialauth.h"
#include <security/pam_modules.h>
#include <security/pam_ext.h>
#include <syslog.h>
#include <string>
#include <vector>
#include <cstring>

// Funzione per caricare la configurazione dal file usando la funzione esistente
extern "C" {
    PAM_EXTERN int pam_sm_authenticate(
        pam_handle_t *pamh,
        int flags,
        int argc,
        const char **argv)
    {
        const char *user = nullptr;
        bool debug_enabled = false;  // Flag per il debug dalla linea di comando
        std::string cfg_err;

        // Verifica se il parametro debug è stato passato dalla linea di comando PAM
        for (int i = 0; i < argc; ++i) {
            if (strcmp(argv[i], "debug") == 0) {
                debug_enabled = true;
                break;
            }
        }

        // Carga la configurazione dal file usando la funzione fa_read_config
        FacialAuthConfig cfg;
        if (!fa_read_config(cfg, cfg_err, "/etc/security/pam_facial.conf")) {
            pam_syslog(pamh, LOG_ERR, "Config load failed: %s", cfg_err.c_str());
            return PAM_AUTH_ERR;
        }

        // Se debug è attivato dalla linea di comando, usa questa configurazione
        if (debug_enabled) {
            cfg.debug = true;
        }

        // Inizia il logging dettagliato se debug è attivo
        if (cfg.debug) {
            openlog("pam_facial_auth", LOG_PID | LOG_CONS, LOG_USER);
            syslog(LOG_DEBUG, "Debug mode enabled: Detailed logs will be written.");
        } else {
            openlog("pam_facial_auth", LOG_PID | LOG_CONS, LOG_USER);
            syslog(LOG_INFO, "Standard log: Only essential information.");
        }

        // Restante codice di autenticazione PAM
        if (pam_get_user(pamh, &user, nullptr) != PAM_SUCCESS || !user) {
            pam_syslog(pamh, LOG_ERR, "Cannot get PAM user");
            return PAM_AUTH_ERR;
        }

        // Loggare un messaggio di debug se necessario
        if (cfg.debug) {
            syslog(LOG_DEBUG, "Debug: Authenticating user %s", user);
        }

        // Loggare informazioni essenziali
        syslog(LOG_INFO, "Authenticating user %s", user);

        return PAM_SUCCESS;
    }
}
