#ifndef FACIALAUTH_H
#define FACIALAUTH_H

#include "Utils.h"
#include "FaceRecWrapper.h"
#include <security/pam_appl.h>
#include <string>

namespace FacialAuth {

    // percorso file config di default
    inline constexpr const char* DEFAULT_CONF = "/etc/pam_facial_auth/pam_facial.conf";

    // carica configurazione (file + eventuali override argv stile PAM/CLI)
    void load_config(FacialAuthConfig &cfg, const char **argv=nullptr, int argc=0, std::string *trace=nullptr);

    // costruisce percorso modello utente e verifica esistenza
    // out: model_base (senza estensione), model_file (quello trovato)
    bool resolve_user_model(const FacialAuthConfig &cfg, const std::string &user,
                            std::string &model_base, std::string &model_file);

    // ciclo di riconoscimento (bloccante) con timeout; ritorna true se recognized
    bool recognize_loop(const FacialAuthConfig &cfg, const std::string &user,
                        bool verbose_to_pam, pam_handle_t *pamh,
                        double &out_conf);

    // training automatico da camera: cattura N frame dal volto, addestra e salva
    // ritorna true se il modello è stato salvato correttamente
    bool auto_train_from_camera(const FacialAuthConfig &cfg, const std::string &user);

}

#endif


