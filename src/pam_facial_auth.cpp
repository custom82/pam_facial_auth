#include "../include/libfacialauth.h"

#include <security/pam_modules.h>
#include <security/pam_ext.h>

#include <syslog.h>
#include <pwd.h>
#include <unistd.h>
#include <cstring>
#include <string>

// Semplice helper per loggare su syslog con prefisso [pam_facial_auth]
static void pam_log(int priority, const std::string &msg)
{
    syslog(priority, "[pam_facial_auth] %s", msg.c_str());
}

extern "C" {

    // =====================================================================
    // pam_sm_authenticate
    // =====================================================================
    PAM_EXTERN int pam_sm_authenticate(
        pam_handle_t *pamh,
        int flags,
        int argc,
        const char **argv
    )
    {
        (void)flags;

        openlog("pam_facial_auth", LOG_PID, LOG_AUTHPRIV);

        // --------------------------------------------------------------
        // Recupera utente da PAM
        // --------------------------------------------------------------
        const char *user_c = nullptr;
        int pret = pam_get_user(pamh, &user_c, "Username: ");
        if (pret != PAM_SUCCESS || !user_c || !*user_c) {
            pam_log(LOG_ERR, "Failed to get PAM user");
            closelog();
            return PAM_AUTH_ERR;
        }

        std::string user(user_c);

        // --------------------------------------------------------------
        // Opzioni PAM:
        //   debug
        //   config=/path/to/pam_facial.conf
        //   cuda=true|false|yes|no|1|0
        //   opencl=true|false|yes|no|1|0
        // --------------------------------------------------------------
        bool cli_debug   = false;
        bool cli_cuda    = false;
        bool cli_opencl  = false;
        std::string config_path = FACIALAUTH_CONFIG_DEFAULT;

        auto parse_bool = [](const char *v) -> bool {
            if (!v) return false;
            if (strcmp(v, "1") == 0) return true;
            if (strcasecmp(v, "true") == 0) return true;
            if (strcasecmp(v, "yes") == 0)  return true;
            return false;
        };

        for (int i = 0; i < argc; ++i) {
            if (!argv[i])
                continue;

            if (strcmp(argv[i], "debug") == 0) {
                cli_debug = true;
            } else if (strncmp(argv[i], "config=", 7) == 0) {
                config_path = std::string(argv[i] + 7);
            } else if (strncmp(argv[i], "cuda=", 5) == 0) {
                cli_cuda = parse_bool(argv[i] + 5);
            } else if (strncmp(argv[i], "opencl=", 7) == 0) {
                cli_opencl = parse_bool(argv[i] + 7);
            }
        }

        // --------------------------------------------------------------
        // Carica configurazione
        // --------------------------------------------------------------
        FacialAuthConfig cfg;
        std::string logbuf;

        if (!fa_load_config(cfg, logbuf, config_path)) {
            pam_log(LOG_ERR,
                    "Cannot load config file: " + config_path);
            if (!logbuf.empty())
                pam_log(LOG_ERR, logbuf);
            closelog();
            return PAM_AUTH_ERR;
        }

        // debug da PAM > config
        if (cli_debug)
            cfg.debug = true;

        if (cfg.debug && !logbuf.empty()) {
            pam_log(LOG_DEBUG,
                    "Config loaded from " + config_path + ":\n" + logbuf);
        }

        // --------------------------------------------------------------
        // Override CUDA / OPENCL da PAM, rispettando i macro di build
        // --------------------------------------------------------------
        #ifdef ENABLE_CUDA
        if (cli_cuda) {
            cfg.dnn_backend = "cuda";
            cfg.dnn_target  = "cuda";
            if (cfg.debug)
                pam_log(LOG_DEBUG, "PAM option cuda=true: forcing DNN backend/target to CUDA");
        }
        #else
        if (cli_cuda && cfg.debug) {
            pam_log(LOG_DEBUG,
                    "PAM option cuda=true requested but pam_facial_auth was built without ENABLE_CUDA, ignoring");
        }
        #endif

        #ifdef ENABLE_OPENCL
        if (cli_opencl) {
            cfg.dnn_backend = "opencl";
            cfg.dnn_target  = "opencl";
            if (cfg.debug)
                pam_log(LOG_DEBUG, "PAM option opencl=true: forcing DNN backend/target to OpenCL");
        }
        #else
        if (cli_opencl && cfg.debug) {
            pam_log(LOG_DEBUG,
                    "PAM option opencl=true requested but pam_facial_auth was built without ENABLE_OPENCL, ignoring");
        }
        #endif

        // --------------------------------------------------------------
        // Costruisci path modello utente
        // --------------------------------------------------------------
        std::string modelPath = fa_user_model_path(cfg, user);

        if (cfg.debug) {
            pam_log(LOG_DEBUG,
                    "Authenticating user '" + user + "' with model '" + modelPath + "'" +
                    ", backend='" + cfg.dnn_backend + "', target='" + cfg.dnn_target + "'");
        }

        // --------------------------------------------------------------
        // Esegui test
        // --------------------------------------------------------------
        double best_conf = 0.0;
        int best_label   = -1;
        std::string logbuf_test;

        // threshold_override < 0 → usa cfg.sface_threshold / soglie classiche
        bool ok = fa_test_user(
            user,
            cfg,
            modelPath,
            best_conf,
            best_label,
            logbuf_test,
            -1.0
        );

        if (!logbuf_test.empty()) {
            pam_log(cfg.debug ? LOG_DEBUG : LOG_INFO, logbuf_test);
        }

        if (ok) {
            pam_log(LOG_INFO,
                    "Authentication SUCCESS for user '" + user +
                    "', similarity/confidence=" + std::to_string(best_conf));
            closelog();
            return PAM_SUCCESS;
        } else {
            pam_log(LOG_INFO,
                    "Authentication FAILED for user '" + user +
                    "', best similarity/confidence=" + std::to_string(best_conf));
            closelog();
            return PAM_AUTH_ERR;
        }
    }

    // =====================================================================
    // pam_sm_setcred: no-op
    // =====================================================================
    PAM_EXTERN int pam_sm_setcred(
        pam_handle_t *pamh,
        int flags,
        int argc,
        const char **argv
    )
    {
        (void)pamh;
        (void)flags;
        (void)argc;
        (void)argv;
        return PAM_SUCCESS;
    }

} // extern "C"
