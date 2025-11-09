#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>  // Required for FaceRecognizer

class FaceRecWrapper {
public:
    FaceRecWrapper();
    FaceRecWrapper(const std::string &techniqueName, const std::string &pathCascade);

    void Train(const std::vector<cv::Mat> &images, const std::vector<int> &labels);
    void Predict(const cv::Mat &im, int &label, double &confidence);
    void Save(const std::string &path);
    void Load(const std::string &path);

    void SetLabelNames(const std::vector<std::string> &names);  // Declare SetLabelNames
    std::string GetLabelName(int index);  // Declare GetLabelName

private:
    cv::Ptr<cv::face::FaceRecognizer> fr;
    cv::CascadeClassifier cascade;
    std::vector<std::string> labelNames;  // Store the label names

    bool SetTechnique(const std::string &t);
    bool LoadCascade(const std::string &filepath);
    bool CropFace(const cv::Mat &image, cv::Mat &cropped);

    std::string pathCascade;
    int sizeFace;
    std::string technique;
};
