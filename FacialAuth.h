// FacialAuth.h
#ifndef FACIALAUTH_H
#define FACIALAUTH_H

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <string>

class FacialAuth {
public:
    FacialAuth();
    bool authenticate(const cv::Mat &image);
    // Aggiungi altre dichiarazioni di metodi che vuoi utilizzare
private:
    // Aggiungi eventuali variabili membro necessarie
};

#endif // FACIALAUTH_H

