#define PAM_SM_AUTH

#include "libfacialauth.h"

#include <security/pam_appl.h>
#include <security/pam_modules.h>
#include <security/pam_ext.h>

#include <sys/stat.h>
#include <syslog.h>

#include <cstring>
#include <string>

using std::string;

static bool local_file_exists(const string &path) {
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

static bool arg_is(const char *arg, const char *name) {
    return std::strcmp(arg, name) == 0;
}

extern "C" {

PAM_EXTERN int pam_sm_authenticate(pam_handle_t *pamh, int flags,
                                   int argc, const char **argv) {
    (void)flags;

    const char *user_c = nullptr;
    int pret = pam_get_user(pamh, &user_c, nullptr);
    if (pret != PAM_SUCCESS || !user_c || !*user_c) {
        pam_syslog(pamh, LOG_ERR, "pam_facial_auth: pam_get_user failed");
        return PAM_AUTH_ERR;
    }
    string user(user_c);

    string cfg_path = FACIALAUTH_CONFIG_DEFAULT;
    bool debug_override = false;

    for (int i = 0; i < argc; ++i) {
        const char *arg = argv[i];
        if (!arg) continue;
        if (std::strncmp(arg, "config=", 7) == 0) {
            cfg_path = string(arg + 7);
        } else if (arg_is(arg, "debug") || arg_is(arg, "debug=1") || arg_is(arg, "debug=true")) {
            debug_override = true;
        }
    }

    if (!local_file_exists(cfg_path)) {
        pam_syslog(pamh, LOG_ERR, "pam_facial_auth: config file '%s' not found", cfg_path.c_str());
        return PAM_AUTH_ERR;
    }

    FacialAuthConfig cfg;
    string err;
    if (!fa_read_config(cfg_path, cfg, err)) {
        pam_syslog(pamh, LOG_ERR, "pam_facial_auth: error reading config '%s': %s",
                   cfg_path.c_str(), err.c_str());
        return PAM_AUTH_ERR;
    }
    if (debug_override) cfg.debug = true;

    double best_conf = 0.0;
    string method;
    string err2;

    bool ok = fa_test_user(cfg, user, best_conf, method, err2);

    if (!ok) {
        pam_syslog(pamh, LOG_ERR,
                   "pam_facial_auth: auth FAILED for user=%s method=%s conf=%.3f: %s",
                   user.c_str(), method.c_str(), best_conf, err2.c_str());
        return PAM_AUTH_ERR;
    }

    pam_syslog(pamh, LOG_INFO,
               "pam_facial_auth: auth SUCCESS for user=%s method=%s conf=%.3f",
               user.c_str(), method.c_str(), best_conf);

    return PAM_SUCCESS;
}

PAM_EXTERN int pam_sm_setcred(pam_handle_t *pamh, int flags, int argc, const char **argv) {
    (void)pamh;
    (void)flags;
    (void)argc;
    (void)argv;
    return PAM_SUCCESS;
}

} // extern "C"
