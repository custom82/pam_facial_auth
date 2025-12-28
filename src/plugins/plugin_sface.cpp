#include "libfacialauth.h"
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/face.hpp>
#include <fstream>

class SFacePlugin : public RecognizerPlugin {
    cv::Ptr<cv::FaceRecognizerSF> sface;
    cv::Mat embeddings; // NxD CV_32F

public:
    SFacePlugin(const FacialAuthConfig& cfg) {
        // modello ONNX SFace
        sface = cv::FaceRecognizerSF::create(cfg.recognize_sface, "");
    }

    std::string get_name() const override { return "sface"; }

    bool load(const std::string& path) override {
        cv::FileStorage fs(path, cv::FileStorage::READ);
        if (!fs.isOpened()) return false;

        cv::FileNode h = fs["pfa_header"];
        if (h.empty()) return false;

        std::string alg;
        h["algorithm"] >> alg;
        if (alg != "sface") return false;

        fs["embeddings"] >> embeddings;
        return !embeddings.empty() && embeddings.type() == CV_32F;
    }

    bool train(const std::vector<cv::Mat>& faces,
               const std::vector<int>& /*labels*/,
               const std::string& save_path) override
               {
                   // genera embeddings: 1 per immagine
                   std::vector<cv::Mat> vec;
                   vec.reserve(faces.size());

                   for (const auto& img : faces) {
                       if (img.empty()) continue;

                       cv::Mat face = img;
                       // SFace lavora bene con facce "piccole" uniformi. Se non hai alignment,
                       // almeno ridimensiona a 112x112.
                       if (face.cols != 112 || face.rows != 112)
                           cv::resize(face, face, cv::Size(112,112));

                       cv::Mat emb;
                       sface->feature(face, emb);   // emb: 1xD CV_32F
                       if (emb.empty() || emb.type() != CV_32F) return false;

                       vec.push_back(emb.clone());
                   }

                   if (vec.empty()) return false;

                   // concat in NxD
                   embeddings = cv::Mat((int)vec.size(), vec[0].cols, CV_32F);
                   for (int i = 0; i < (int)vec.size(); ++i) {
                       vec[i].copyTo(embeddings.row(i));
                   }

                   // Scrivi XML con header + embeddings
                   cv::FileStorage fs(save_path, cv::FileStorage::WRITE);
                   if (!fs.isOpened()) return false;

                   fs << "pfa_header" << "{"
                   << "version" << 1
                   << "algorithm" << "sface"
                   << "embedding_dim" << embeddings.cols
                   << "embedding_count" << embeddings.rows
                   << "}";

                   fs << "embeddings" << embeddings;
                   return true;
               }

               bool predict(const cv::Mat& face, int& label, double& confidence) override {
                   if (embeddings.empty()) return false;

                   cv::Mat f = face;
                   if (f.cols != 112 || f.rows != 112)
                       cv::resize(f, f, cv::Size(112,112));

                   cv::Mat emb;
                   sface->feature(f, emb);
                   if (emb.empty() || emb.type() != CV_32F) return false;

                   // match best cosine
                   double best = -1e9;
                   for (int i = 0; i < embeddings.rows; ++i) {
                       double sim = sface->match(emb, embeddings.row(i), cv::FaceRecognizerSF::FR_COSINE);
                       if (sim > best) best = sim;
                   }

                   // label "1" se c’è un confronto valido; confidence = similarità
                   label = 1;
                   confidence = best;
                   return true;
               }
};
