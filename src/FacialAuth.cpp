#include <security/pam_appl.h>  // Aggiungi questa riga
#include "FaceRecWrapper.h"

int pam_sm_authenticate(pam_handle_t *pamh, int flags, int argc, const char **argv) {
	// Logica per l'autenticazione facciale
	// Ad esempio, usa FaceRecWrapper per fare la previsione
	cv::Mat image = ... // Acquisisci immagine dalla webcam
	int prediction;
	double confidence;
	FaceRecWrapper frw;
	frw.Predict(image, prediction, confidence);

	// Esegui il controllo sulla previsione
	const std::string username = "test_user"; // Sostituisci con il nome utente corretto
	if (confidence < threshold && frw.GetLabelName(prediction) == username) {
		// Autenticazione riuscita
		return PAM_SUCCESS;
	} else {
		// Autenticazione fallita
		return PAM_AUTH_ERR;
	}
}
