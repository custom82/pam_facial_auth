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
        if (!cfg.recognize_sface.empty()) {
            sface = cv::FaceRecognizerSF::create(cfg.recognize_sface, "");
        }
    }

    std::string get_name() const override { return "sface"; }

    bool load(const std::string& path, std::string& err) override {
        cv::FileStorage fs(path, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            err = "Impossibile aprire il modello: " + path;
            return false;
        }

        cv::FileNode h = fs["pfa_header"];
        if (h.empty()) {
            err = "Header modello mancante";
            return false;
        }

        std::string alg;
        h["algorithm"] >> alg;
        if (alg != "sface") {
            err = "Algoritmo non compatibile: " + alg;
            return false;
        }

        fs["embeddings"] >> embeddings;
        if (embeddings.empty() || embeddings.type() != CV_32F) {
            err = "Embeddings non validi nel modello";
            return false;
        }
        return true;
    }

    bool train(const std::vector<cv::Mat>& faces,
               const std::vector<int>& /*labels*/,
               const std::string& save_path,
               std::string& err) override
               {
                   if (sface.empty()) {
                       err = "Modello SFace non inizializzato (recognize_sface mancante?)";
                       return false;
                   }

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
                       if (emb.empty() || emb.type() != CV_32F) {
                           err = "Embedding non valido";
                           return false;
                       }

                       vec.push_back(emb.clone());
                   }

                   if (vec.empty()) {
                       err = "Nessuna immagine valida per il training";
                       return false;
                   }

                   // concat in NxD
                   embeddings = cv::Mat((int)vec.size(), vec[0].cols, CV_32F);
                   for (int i = 0; i < (int)vec.size(); ++i) {
                       vec[i].copyTo(embeddings.row(i));
                   }

                   // Scrivi XML con header + embeddings
                   cv::FileStorage fs(save_path, cv::FileStorage::WRITE);
                   if (!fs.isOpened()) {
                       err = "Impossibile salvare il modello: " + save_path;
                       return false;
                   }

                   fs << "pfa_header" << "{"
                   << "version" << 1
                   << "algorithm" << "sface"
                   << "embedding_dim" << embeddings.cols
                   << "embedding_count" << embeddings.rows
                   << "}";

                   fs << "embeddings" << embeddings;
                   return true;
               }

               bool predict(const cv::Mat& face, int& label, double& confidence, std::string& err) override {
                   if (embeddings.empty()) {
                       err = "Embeddings non caricati";
                       return false;
                   }
                   if (sface.empty()) {
                       err = "Modello SFace non inizializzato";
                       return false;
                   }

                   cv::Mat f = face;
                   if (f.cols != 112 || f.rows != 112)
                       cv::resize(f, f, cv::Size(112,112));

                   cv::Mat emb;
                   sface->feature(f, emb);
                   if (emb.empty() || emb.type() != CV_32F) {
                       err = "Embedding non valido";
                       return false;
                   }

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

               bool is_match(double confidence, const FacialAuthConfig& cfg) const override {
                   return confidence >= cfg.sface_threshold;
               }
};

std::unique_ptr<RecognizerPlugin> create_sface_plugin(const FacialAuthConfig& cfg) {
    return std::make_unique<SFacePlugin>(cfg);
}
