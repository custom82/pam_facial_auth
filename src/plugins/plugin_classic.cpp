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

    bool load(const std::string& path, std::string& err) override {
        cv::FileStorage fs(path, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            err = "Impossibile aprire il modello: " + path;
            return false;
        }

        cv::FileNode h = fs["pfa_header"];
        if (!h.empty()) {
            std::string alg;
            h["algorithm"] >> alg;
            if (!alg.empty() && alg != type) {
                err = "Modello non compatibile (atteso " + type + ", trovato " + alg + ")";
                return false;
            }
        }

        // il read() ignora i nodi extra e prende i suoi
        model->read(path);
        return true;
    }

    bool train(const std::vector<cv::Mat>& faces,
               const std::vector<int>& labels,
               const std::string& save_path,
               std::string& err) override
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
                   if (gray.empty()) {
                       err = "Nessuna immagine valida per il training";
                       return false;
                   }

                   model->train(gray, labels);

                   cv::FileStorage fs(save_path, cv::FileStorage::WRITE);
                   if (!fs.isOpened()) {
                       err = "Impossibile salvare il modello: " + save_path;
                       return false;
                   }

                   // Header PFA
                   fs << "pfa_header" << "{"
                   << "version" << 1
                   << "algorithm" << type
                   << "}";

                   // Dati del riconoscitore OpenCV (LBPH/Eigen/Fisher)
                   model->write(fs);
                   return true;
               }

               bool predict(const cv::Mat& face, int& label, double& confidence, std::string& err) override {
                   cv::Mat g;
                   if (face.empty()) {
                       err = "Immagine vuota";
                       return false;
                   }
                   if (face.channels() == 3) cv::cvtColor(face, g, cv::COLOR_BGR2GRAY);
                   else g = face;
                   model->predict(g, label, confidence);
                   return true;
               }

               bool is_match(double confidence, const FacialAuthConfig& cfg) const override {
                   if (type == "lbph") return confidence <= cfg.lbph_threshold;
                   if (type == "eigen") return confidence <= cfg.eigen_threshold;
                   if (type == "fisher") return confidence <= cfg.fisher_threshold;
                   return confidence <= cfg.threshold;
               }
};

std::unique_ptr<RecognizerPlugin> create_classic_plugin(const std::string& method, const FacialAuthConfig& cfg) {
    return std::make_unique<ClassicPlugin>(method, cfg);
}
