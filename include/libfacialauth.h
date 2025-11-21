#ifndef LIBFACIALAUTH_H
#define LIBFACIALAUTH_H

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <opencv2/dnn.hpp>

// ==========================================================
// Default config file path
// ==========================================================
#ifndef FACIALAUTH_CONFIG_DEFAULT
#define FACIALAUTH_CONFIG_DEFAULT "/etc/security/pam_facial.conf"
#endif

// ==========================================================
// Configurazione globale
// ==========================================================

struct FacialAuthConfig {
    // Path al file di configurazione (solo informativo)
    std::string config_path = FACIALAUTH_CONFIG_DEFAULT;

    // Directory base per immagini e modelli
    //  images: <basedir>/images/<user>/
    //  models: <basedir>/models/<user>.xml
    std::string basedir = "/etc/pam_facial_auth";

    // Dispositivo video (se vuoto, usa camera_index)
    std::string device;

    // Indice camera (se device vuoto)
    int camera_index = 0;

    // Numero di frame da catturare per test/train
    int frames = 10;

    // Dimensioni standard del volto normalizzato
    int width  = 200;
    int height = 200;

    // Millisecondi di sleep tra un frame e l'altro
    int sleep_ms = 150;

    // Soglia decisionale per metodi classici (LBPH/Eigen/Fisher)
    // e per DNN (dove viene trasformata).
    double threshold = 60.0;

    // Flag vari
    bool debug           = false;
    bool force_overwrite = false;

    // ======================================================
    // Parametri DNN (validi per tutti i backend)
    // ======================================================
    // "caffe", "tensorflow", "onnx", "openvino"
    std::string dnn_type = "caffe";

    // path al file modello (caffemodel, .pb, .onnx, .xml)
    std::string dnn_model_path;

    // path al file di “config” (prototxt, pbtxt, .bin IR, opzionale)
    std::string dnn_proto_path;

    // backend/target: "cpu", "cuda", "opencl", "openvino"
    std::string dnn_device = "cpu";

    // soglia logica [0–1], es: 0.6
    double dnn_threshold = 0.6;
};

// ==========================================================
// API di configurazione
// ==========================================================

// Carica configurazione semplice key=value dal file path
// Restituisce false in caso di errore grave (file mancante, ecc.)
bool fa_load_config(const std::string &path,
                    FacialAuthConfig &cfg,
                    std::string &log);

// Directory immagini e path modello per un utente
std::string fa_user_image_dir(const FacialAuthConfig &cfg,
                              const std::string &user);

std::string fa_user_model_path(const FacialAuthConfig &cfg,
                               const std::string &user);

// ==========================================================
// Wrapper di riconoscimento
// ==========================================================

class FaceRecWrapper {
public:
    FaceRecWrapper();
    explicit FaceRecWrapper(const std::string &modelType_);

    bool Load(const std::string &modelFile);
    bool Save(const std::string &modelFile) const;

    bool Train(const std::vector<cv::Mat> &images,
               const std::vector<int> &labels);

    bool Predict(const cv::Mat &faceGray,
                 int &label,
                 double &confidence);

    bool DetectFace(const cv::Mat &frame, cv::Rect &faceROI);

    // Configura DNN a partire dalla config globale
    void ConfigureDNN(const FacialAuthConfig &cfg);

private:
    std::string modelType; // "lbph","eigen","fisher","dnn"
    cv::Ptr<cv::face::FaceRecognizer> recognizer;

    // --- Stato DNN ---
    bool        use_dnn        = false;
    bool        dnn_loaded     = false;

    std::string dnn_type;
    std::string dnn_model_path;
    std::string dnn_proto_path;
    std::string dnn_device;
    double      dnn_threshold  = 0.6;
    cv::dnn::Net dnn_net;

    // Stato detector (Haar Cascade)
    mutable cv::CascadeClassifier faceCascade;

    // Helpers interni
    void init_recognizer_for_algorithm(const std::string &algo);
    bool load_dnn_from_model_file(const std::string &modelFile);
    bool predict_with_dnn(const cv::Mat &faceGray,
                          int &label,
                          double &confidence);
};

// ==========================================================
// API alto livello
// ==========================================================

// Cattura immagini da webcam e salva in basedir/images/<user>/img_XXX.<fmt>
bool fa_capture_images(const std::string &user,
                       const FacialAuthConfig &cfg,
                       bool force,
                       std::string &log,
                       const std::string &img_format = "png");

// Allena un modello per l’utente (LBPH / Eigen / Fisher / DNN)
// - method: "lbph", "eigen", "fisher", "dnn"
// - inputDir: se vuoto usa fa_user_image_dir(cfg, user)
// - outputModel: se vuoto usa fa_user_model_path(cfg, user)
bool fa_train_user(const std::string &user,
                   const FacialAuthConfig &cfg,
                   const std::string &method,
                   const std::string &inputDir,
                   const std::string &outputModel,
                   bool force,
                   std::string &log);

// Testa l’utente in tempo reale (camera) usando il modello XML
// - modelPath: se vuoto usa fa_user_model_path(cfg, user)
// - best_conf: migliore (più bassa) conf trovata
// - best_label: label associato
bool fa_test_user(const std::string &user,
                  const FacialAuthConfig &cfg,
                  const std::string &modelPath,
                  double &best_conf,
                  int &best_label,
                  std::string &log);

// ==========================================================
// Maintenance helpers and root check
// ==========================================================
bool fa_clean_images(const FacialAuthConfig &cfg, const std::string &user);
bool fa_clean_model(const FacialAuthConfig &cfg, const std::string &user);
void fa_list_images(const FacialAuthConfig &cfg, const std::string &user);
bool fa_check_root(const char *tool_name);

#endif // LIBFACIALAUTH_H
