#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <map>
#include <string>

class Utils {
public:
    static bool GetConfig(const std::string& configPath, std::map<std::string, std::string>& config);
    static cv::Mat LoadImage(const std::string& imagePath);
    static cv::Mat ResizeImage(const cv::Mat& image, int width, int height);
    static void ShowImage(const cv::Mat& image, const std::string& windowName);
    static void ProcessImage(const cv::Mat& image);
};

#endif // UTILS_H

