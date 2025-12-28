#include "libfacialauth.h"

#include <opencv2/face.hpp>
#include <opencv2/imgproc.hpp>

class ClassicPlugin : public RecognizerPlugin {
    cv::Ptr<cv::face::FaceRecognizer> model;
    std::string type;

public:
    ClassicPlugin(const std::string& method, const FacialAuthConfig& /*cfg*/) : type(method) {
        if (method == "lbph") model = cv::face::LBPHFaceRecognizer::create();
        else if (method == "eigen") model = cv::face::EigenFaceRecognizer::create();
        else model = cv::face::FisherFaceRecognizer::create();
    }

    std::string get_name() const override { return type; }

    bool load(const std::string& path, std::string& err) override {
        cv::FileStorage fs(path, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            err = "Unable to open model " + path;
            return false;
        }

        cv::FileNode header = fs["pfa_header"];
        if (header.empty()) {
            err = "Missing model header: " + path;
            return false;
        }

        std::string alg;
        header["algorithm"] >> alg;
        if (!alg.empty() && alg != type) {
            err = "Model algorithm " + alg + " is incompatible with " + type;
            return false;
        }

        // read() ignores extra nodes and reads its own.
        model->read(path);
        return true;
    }

    bool train(const std::vector<cv::Mat>& faces,
               const std::vector<int>& labels,
               const std::string& save_path,
               std::string& err) override {
        std::vector<cv::Mat> gray;
        gray.reserve(faces.size());
        for (auto& f : faces) {
            if (f.empty()) continue;
            cv::Mat g;
            if (f.channels() == 3) cv::cvtColor(f, g, cv::COLOR_BGR2GRAY);
            else g = f;
            gray.push_back(g);
        }
        if (gray.empty()) {
            err = "No valid images for training";
            return false;
        }

        model->train(gray, labels);

        cv::FileStorage fs(save_path, cv::FileStorage::WRITE);
        if (!fs.isOpened()) {
            err = "Unable to write model: " + save_path;
            return false;
        }

        // PFA header
        fs << "pfa_header" << "{"
           << "version" << 1
           << "algorithm" << type
           << "}";

        // OpenCV recognizer data (LBPH/Eigen/Fisher)
        model->write(fs);
        return true;
    }

    bool predict(const cv::Mat& face, int& label, double& confidence, std::string& err) override {
        if (face.empty()) {
            err = "Empty face for predict";
            return false;
        }
        cv::Mat g;
        if (face.channels() == 3) cv::cvtColor(face, g, cv::COLOR_BGR2GRAY);
        else g = face;
        model->predict(g, label, confidence);
        return true;
    }

    bool is_match(double confidence, const FacialAuthConfig& cfg) const override {
        double threshold = cfg.threshold;
        if (type == "lbph") threshold = cfg.lbph_threshold;
        else if (type == "eigen") threshold = cfg.eigen_threshold;
        else if (type == "fisher") threshold = cfg.fisher_threshold;
        return confidence <= threshold;
    }
};

std::unique_ptr<RecognizerPlugin> create_classic_plugin(const std::string& method, const FacialAuthConfig& cfg) {
    return std::make_unique<ClassicPlugin>(method, cfg);
}
