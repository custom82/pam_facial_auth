#ifndef PAM_FACIAL_AUTH_HPP
#define PAM_FACIAL_AUTH_HPP

#include <security/pam_appl.h>
#include <security/pam_misc.h>
#include <security/pam_modules.h>
#include <security/pam_ext.h>
#include <opencv2/opencv.hpp>
#include "FacialAuth.h"  // Include the FacialAuth header for authentication class
#include <syslog.h>

class PAMFacialAuth {
public:
    PAMFacialAuth();
    ~PAMFacialAuth();

    // Authenticate user based on the facial recognition
    int authenticate(pam_handle_t *pamh, int flags, int argc, const char **argv);

private:
    // Helper method to retrieve the user from the PAM handle
    const char* getUser(pam_handle_t *pamh);

    // PAM status variable
    int retval;
};

#endif // PAM_FACIAL_AUTH_HPP
