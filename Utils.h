#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <map>
#include <string>
#include <vector>

class Utils {
public:
    static bool GetConfig(const std::string& configPath, std::map<std::string, std::string>& config);
    static cv::Mat LoadImage(const std::string& imagePath);
    static cv::Mat ResizeImage(const cv::Mat& image, int width, int height);
    static void ShowImage(const cv::Mat& image, const std::string& windowName);
    static void ProcessImage(const cv::Mat& image);

    // Dichiarazione della funzione WalkDirectory
    static void WalkDirectory(const std::string& dirPath, std::vector<std::string>& fileNames, std::vector<std::string>& userNames);
};

#endif // UTILS_H

