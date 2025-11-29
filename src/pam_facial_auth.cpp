#include <security/pam_modules.h>
#include <security/pam_ext.h>
#include <security/pam_appl.h>

#include <string>
#include <cstring>
#include <cctype>

#include <sys/stat.h>   // <-- necessario per stat, S_ISREG
#include <syslog.h>     // <-- necessario per LOG_ERR, LOG_INFO

#include "../include/libfacialauth.h"

// =====================================================================
// Helpers locali
// =====================================================================

// tiny str_to_bool locale
static bool local_str_to_bool(const char *s, bool defval)
{
    if (!s) return defval;

    std::string t = s;
    for (char &c : t)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    if (t == "1" || t == "true"  || t == "yes" || t == "on")  return true;
    if (t == "0" || t == "false" || t == "no"  || t == "off") return false;

    return defval;
}

// tiny file_exists locale
static bool local_file_exists(const std::string &path)
{
    struct stat st {};
    return (::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode));
}

// =====================================================================
// PAM Auth
// =====================================================================

extern "C" {

    PAM_EXTERN int pam_sm_authenticate(pam_handle_t *pamh,
                                       int flags,
                                       int argc,
                                       const char **argv)
    {
        (void)flags;

        const char *user_c = nullptr;
        if (pam_get_user(pamh, &user_c, nullptr) != PAM_SUCCESS || !user_c)
            return PAM_AUTH_ERR;

        std::string user(user_c);

        // -----------------------------------------------------
        // Config + override
        // -----------------------------------------------------
        FacialAuthConfig cfg;
        std::string cfg_path = FACIALAUTH_CONFIG_DEFAULT;

        bool debug_override = false;

        for (int i = 0; i < argc; ++i) {
            if (!argv[i]) continue;

            if (strncmp(argv[i], "config=", 7) == 0)
                cfg_path = std::string(argv[i] + 7);

            else if (strncmp(argv[i], "debug=", 6) == 0)
                debug_override = local_str_to_bool(argv[i] + 6, false);
        }

        std::string logbuf;
        read_kv_config(cfg_path, cfg, &logbuf);

        if (debug_override)
            cfg.debug = true;

        // -----------------------------------------------------
        // Percorso modello utente
        // -----------------------------------------------------
        std::string model_path = fa_user_model_path(cfg, user);
        if (!local_file_exists(model_path)) {
            pam_syslog(pamh, LOG_ERR,
                       "[pam_facial_auth] model not found: %s",
                       model_path.c_str());
            return PAM_AUTH_ERR;
        }

        // -----------------------------------------------------
        // Test
        // -----------------------------------------------------
        double confidence = 9999.0;

        bool ok = fa_test(user, cfg, confidence, logbuf);

        if (!ok) {
            pam_syslog(pamh, LOG_ERR,
                       "[pam_facial_auth] auth FAILED user=%s conf=%.3f",
                       user.c_str(), confidence);
            return PAM_AUTH_ERR;
        }

        pam_syslog(pamh, LOG_INFO,
                   "[pam_facial_auth] auth OK user=%s conf=%.3f",
                   user.c_str(), confidence);

        return PAM_SUCCESS;
    }

    PAM_EXTERN int pam_sm_setcred(pam_handle_t *pamh,
                                  int flags,
                                  int argc,
                                  const char **argv)
    {
        (void)pamh;
        (void)flags;
        (void)argc;
        (void)argv;
        return PAM_SUCCESS;
    }

} // extern "C"

