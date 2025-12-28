#include "libfacialauth.h"
#include <opencv2/face.hpp>
#include <opencv2/imgproc.hpp>

class ClassicPlugin : public RecognizerPlugin {
    cv::Ptr<cv::face::FaceRecognizer> model;
    std::string type;

public:
    ClassicPlugin(const std::string& method, const FacialAuthConfig& /*cfg*/) : type(method) {
        if (method == "lbph")      model = cv::face::LBPHFaceRecognizer::create();
        else if (method == "eigen")  model = cv::face::EigenFaceRecognizer::create();
        else                        model = cv::face::FisherFaceRecognizer::create();
    }

    std::string get_name() const override { return type; }

    bool load(const std::string& path) override {
        // il read() ignora i nodi extra e prende i suoi
        model->read(path);
        return true;
    }

    bool train(const std::vector<cv::Mat>& faces,
               const std::vector<int>& labels,
               const std::string& save_path) override
               {
                   std::vector<cv::Mat> gray;
                   gray.reserve(faces.size());
                   for (auto& f : faces) {
                       if (f.empty()) continue;
                       cv::Mat g;
                       if (f.channels() == 3) cv::cvtColor(f, g, cv::COLOR_BGR2GRAY);
                       else g = f;
                       gray.push_back(g);
                   }
                   if (gray.empty()) return false;

                   model->train(gray, labels);

                   cv::FileStorage fs(save_path, cv::FileStorage::WRITE);
                   if (!fs.isOpened()) return false;

                   // Header PFA
                   fs << "pfa_header" << "{"
                   << "version" << 1
                   << "algorithm" << type
                   << "}";

                   // Dati del riconoscitore OpenCV (LBPH/Eigen/Fisher)
                   model->write(fs);
                   return true;
               }

               bool predict(const cv::Mat& face, int& label, double& confidence) override {
                   cv::Mat g;
                   if (face.channels() == 3) cv::cvtColor(face, g, cv::COLOR_BGR2GRAY);
                   else g = face;
                   model->predict(g, label, confidence);
                   return true;
               }
};
