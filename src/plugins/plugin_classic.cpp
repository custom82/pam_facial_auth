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
            threshold = 80.0; // Valore indicativo, usa cfg se preferisci
        } else if (method == "eigen") {
            model = cv::face::EigenFaceRecognizer::create();
            threshold = 5000.0;
        } else {
            model = cv::face::FisherFaceRecognizer::create();
            threshold = 500.0;
        }
    }

    bool load(const std::string& path) override { return model->read(path), true; }

    bool train(const std::vector<cv::Mat>& faces, const std::vector<int>& labels, const std::string& save_path) override {
        std::vector<cv::Mat> grays;
        for(auto& f : faces) {
            cv::Mat g;
            if(f.channels() == 3) cv::cvtColor(f, g, cv::COLOR_BGR2GRAY); else g = f;
            grays.push_back(g);
        }
        model->train(grays, labels);
        model->save(save_path);
        return true;
    }

    bool predict(const cv::Mat& face, int& label, double& confidence) override {
        cv::Mat gray;
        if(face.channels() == 3) cv::cvtColor(face, gray, cv::COLOR_BGR2GRAY); else gray = face;
        model->predict(gray, label, confidence);
        return true;
    }

    std::string get_name() const override { return type; }
};

// FUNZIONE BRIDGE PER IL LINKER
extern "C" std::unique_ptr<RecognizerPlugin> create_classic_plugin(const std::string& method, const FacialAuthConfig& cfg) {
    return std::make_unique<ClassicPlugin>(method, cfg);
}
