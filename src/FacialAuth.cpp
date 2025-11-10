#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <map>
#include <security/pam_appl.h>
#include <security/pam_modules.h>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <ctime>

#include "FaceRecWrapper.h"

// expected hook
PAM_EXTERN int pam_sm_setcred(pam_handle_t *pamh, int flags, int argc, const char **argv) {
	return PAM_SUCCESS;
}

PAM_EXTERN int pam_sm_acct_mgmt(pam_handle_t *pamh, int flags, int argc, const char **argv) {
	return PAM_SUCCESS;
}

// expected hook, this is where custom stuff happens
PAM_EXTERN int pam_sm_authenticate(pam_handle_t *pamh, int flags, int argc, const char **argv) {
	const char *user;
	int ret = pam_get_user(pamh, &user, "Username: ");
	if (ret != PAM_SUCCESS) {
		return ret;
	}
	std::string username(user);

	// Load config
	std::map<std::string, std::string> config;
	Utils::GetConfig("/etc/pam-facial-auth/config", config);

	// Parse config values
	std::time_t timeout = std::stoi(config["timeout"]);
	double threshold = std::stod(config["threshold"]);
	bool imCapture = config["imageCapture"] == "true";

	// Load model
	FaceRecWrapper frw("/etc/pam-facial-auth/model", "face_model");

	std::time_t start = std::time(nullptr);
	std::string imagePathLast;
	cv::VideoCapture vc;
	if (imCapture && !vc.open(0)) {
		printf("Could not open camera.\n");
		return PAM_AUTH_ERR;
	}
	printf("Starting facial recognition for %s...\n", username.c_str());

	while (std::time(nullptr) - start < timeout) {
		cv::Mat im;
		if (imCapture) {
			vc.read(im);
			cv::cvtColor(im, im, cv::COLOR_BGR2GRAY);
		} else {
			// Add image directory walking logic
			// (Assuming `imagePath` is assigned with the correct image)
			im = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
		}

		if (im.empty()) {
			continue;
		}

		int prediction = -1;
		double confidence = 0.0;
		frw.Predict(im, prediction, confidence);

		printf("Predicted: %d, %s (%f)\n", prediction, frw.GetLabelName(prediction).c_str(), confidence);

		if (confidence < threshold && frw.GetLabelName(prediction) == username) {
			return PAM_SUCCESS;
		}
	}

	printf("Timeout on face authentication...\n");
	return PAM_AUTH_ERR;
}
