#include "libfacialauth.h"
#include <opencv2/face.hpp>
#include <opencv2/imgproc.hpp>

class ClassicPlugin : public RecognizerPlugin {
    cv::Ptr<cv::face::FaceRecognizer> model;
    std::string type;
    double threshold;

public:
    ClassicPlugin(const std::string& method, const FacialAuthConfig& cfg) : type(method) {
        if (method == "lbph") {
            model = cv::face::LBPHFaceRecognizer::create();
            threshold = cfg.lbph_threshold;
        } else if (method == "eigen") {
            model = cv::face::EigenFaceRecognizer::create();
            threshold = 5000.0; // Esempio
        } else {
            model = cv::face::FisherFaceRecognizer::create();
            threshold = 500.0; // Esempio
        }
    }

    bool load(const std::string& path) override {
        model->read(path);
        return true;
    }

    bool train(const std::vector<cv::Mat>& faces, const std::vector<int>& labels, const std::string& save_path) override {
        std::vector<cv::Mat> grays;
        for(auto& f : faces) {
            cv::Mat g;
            cv::cvtColor(f, g, cv::COLOR_BGR2GRAY);
            grays.push_back(g);
        }
        model->train(grays, labels);
        model->save(save_path);
        return true;
    }

    bool predict(const cv::Mat& face, int& label, double& confidence) override {
        cv::Mat gray;
        cv::cvtColor(face, gray, cv::COLOR_BGR2GRAY);
        model->predict(gray, label, confidence);
        return (confidence <= threshold); // Per i classici "meno è meglio"
    }

    std::string get_name() const override { return type; }
};
