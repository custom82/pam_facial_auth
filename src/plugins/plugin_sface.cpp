#include "libfacialauth.h"
#include <opencv2/dnn.hpp>
#include <opencv2/objdetect.hpp>
#include <fstream>

class SFacePlugin : public RecognizerPlugin {
    cv::Ptr<cv::FaceRecognizerSF> sface;
    std::vector<cv::Mat> target_embeddings;
    double threshold;

public:
    SFacePlugin(const FacialAuthConfig& cfg) {
        sface = cv::FaceRecognizerSF::create(cfg.recognize_sface, "");
        threshold = cfg.sface_threshold;
    }

    bool load(const std::string& path) override {
        std::ifstream in(path, std::ios::binary);
        if(!in) return false;
        int count; in.read((char*)&count, sizeof(count));
        for(int i=0; i<count; ++i) {
            int r, c, t;
            in.read((char*)&r, sizeof(r)); in.read((char*)&c, sizeof(c)); in.read((char*)&t, sizeof(t));
            cv::Mat m(r, c, t);
            in.read((char*)m.data, m.total() * m.elemSize());
            target_embeddings.push_back(m);
        }
        return !target_embeddings.empty();
    }

    bool train(const std::vector<cv::Mat>& faces, const std::vector<int>& labels, const std::string& save_path) override {
        std::ofstream out(save_path, std::ios::binary);
        int count = faces.size();
        out.write((char*)&count, sizeof(count));
        for (const auto& f : faces) {
            cv::Mat em;
            sface->feature(f, em);
            int r=em.rows, c=em.cols, t=em.type();
            out.write((char*)&r, sizeof(r)); out.write((char*)&c, sizeof(c)); out.write((char*)&t, sizeof(t));
            out.write((char*)em.data, em.total() * em.elemSize());
        }
        return true;
    }

    bool predict(const cv::Mat& face, int& label, double& confidence) override {
        cv::Mat query;
        sface->feature(face, query);
        double max_sim = -1.0;
        for(const auto& target : target_embeddings) {
            double sim = sface->match(query, target, cv::FaceRecognizerSF::DisType::FR_COSINE);
            if (sim > max_sim) max_sim = sim;
        }
        confidence = max_sim;
        label = (confidence >= threshold) ? 0 : -1;
        return (label == 0);
    }

    std::string get_name() const override { return "sface"; }
};
