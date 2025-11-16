#include <security/pam_modules.h>
#include <security/pam_ext.h>
#include <security/pam_appl.h>

#include <string>
#include <cstring>
#include <syslog.h>
#include <iostream>
#include <exception>

#include "../include/libfacialauth.h"

extern "C" {

    PAM_EXTERN int pam_sm_authenticate(pam_handle_t *pamh, int flags,
                                       int argc, const char **argv)
    {
        (void)flags;
        openlog("pam_facial_auth", LOG_PID | LOG_CONS, LOG_AUTHPRIV);
        pam_syslog(pamh, LOG_INFO, "pam_facial_auth: module started");

        FacialAuthConfig cfg;
        std::string config_path = "/etc/security/pam_facial.conf";
        std::string model_path = "/etc/pam_facial_auth/models";
        std::string log;

        try {
            // Legge i parametri PAM
            for (int i = 0; i < argc; ++i) {
                std::string a = argv[i];
                auto pos = a.find('=');
                if (pos == std::string::npos) continue;
                std::string key = a.substr(0, pos);
                std::string val = a.substr(pos + 1);

                if (key == "config")
                    config_path = val;
                else if (key == "threshold")
                    cfg.threshold = std::stod(val);
                else if (key == "device")
                    cfg.device = val;
                else if (key == "nogui")
                    cfg.nogui = str_to_bool(val, cfg.nogui);
                else if (key == "debug")
                    cfg.debug = str_to_bool(val, cfg.debug);
                else if (key == "model")
                    model_path = val;
            }

            pam_syslog(pamh, LOG_INFO, "Reading config: %s", config_path.c_str());
            read_kv_config(config_path, cfg, &log);

            const char *user_c = nullptr;
            int pret = pam_get_user(pamh, &user_c, "Username: ");
            if (pret != PAM_SUCCESS || !user_c || !*user_c) {
                pam_syslog(pamh, LOG_ERR, "Unable to get PAM user");
                closelog();
                return PAM_AUTH_ERR;
            }

            std::string user(user_c);
            std::string user_model_path = model_path + "/" + user + ".xml";

            pam_syslog(pamh, LOG_INFO, "Testing user: %s", user.c_str());
            pam_syslog(pamh, LOG_INFO, "Model path: %s", user_model_path.c_str());
            pam_syslog(pamh, LOG_INFO, "Device: %s", cfg.device.c_str());

            double best_conf = 9999.0;
            int best_label = -1;
            bool ok = fa_test_user(user, cfg, user_model_path, best_conf, best_label, log);

            if (cfg.debug) {
                pam_syslog(pamh, LOG_DEBUG, "pam_facial_auth debug log:\n%s", log.c_str());
                std::cerr << "[pam_facial_auth DEBUG]\n" << log << std::endl;
            }

            if (!ok) {
                pam_syslog(pamh, LOG_NOTICE,
                           "Facial authentication FAILED for user %s (conf=%.2f thr=%.2f)",
                           user.c_str(), best_conf, cfg.threshold);
                closelog();
                return PAM_AUTH_ERR;
            }

            pam_syslog(pamh, LOG_INFO,
                       "Facial authentication SUCCESS for user %s (conf=%.2f <= %.2f)",
                       user.c_str(), best_conf, cfg.threshold);
            closelog();
            return PAM_SUCCESS;

        } catch (const std::exception &e) {
            pam_syslog(pamh, LOG_ERR, "Exception: %s", e.what());
            std::cerr << "[pam_facial_auth EXCEPTION] " << e.what() << std::endl;
            closelog();
            return PAM_AUTH_ERR;
        } catch (...) {
            pam_syslog(pamh, LOG_ERR, "Unknown exception in pam_facial_auth");
            std::cerr << "[pam_facial_auth EXCEPTION] Unknown error\n";
            closelog();
            return PAM_AUTH_ERR;
        }
    }

    PAM_EXTERN int pam_sm_setcred(pam_handle_t *pamh, int flags,
                                  int argc, const char **argv)
    {
        (void)pamh; (void)flags; (void)argc; (void)argv;
        return PAM_SUCCESS;
    }

} // extern "C"
