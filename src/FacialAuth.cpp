#include "FaceRecWrapper.h"

int pam_sm_authenticate(pam_handle_t *pamh, int flags, int argc, const char **argv) {
	// Altri codici...

	int prediction;
	double confidence;

	// Usa il metodo corretto per ottenere l'etichetta
	int result = frw.Predict(image, prediction, confidence);
	if (result == cv::face::FaceRecognizer::ERR_OK) {
		std::string label = frw.GetLabelName(prediction);  // Usa il nome dell'etichetta
		// Verifica se l'etichetta corrisponde al nome dell'utente
		if (confidence < threshold && label == username) {
			// L'autenticazione è riuscita
		}
	}
	return PAM_AUTH_ERR; // Ritorna errore se l'autenticazione non va
}
