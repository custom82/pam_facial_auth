#pragma once

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/videoio.hpp>

// Path di default del file di configurazione
#ifndef FACIALAUTH_CONFIG_DEFAULT
#define FACIALAUTH_CONFIG_DEFAULT "/etc/security/pam_facial.conf"
#endif

// ============================================================
// Struttura di configurazione principale
// ============================================================

struct FacialAuthConfig {
    // Path effettivo del file di config usato
    std::string config_path;

    // Base dir:
    //   images: <basedir>/images/<user>/
    //   models: <basedir>/models/<user>.xml
    std::string basedir = "/etc/pam_facial_auth";

    // Dispositivo camera:
    //   - se device vuoto, usa camera_index (0,1,...)
    //   - altrimenti usa il path in device (es. /dev/video0)
    std::string device;
    int         camera_index = 0;

    // Parametri di cattura / test
    int frames   = 10;   // numero frame da elaborare
    int width    = 200;  // dimensione volto normalizzato (w)
    int height   = 200;  // dimensione volto normalizzato (h)
    int sleep_ms = 150;  // pausa tra i frame

    // Soglia per i metodi classici (LBPH/Eigen/Fisher)
    double threshold = 60.0;

    // Flag di debug e gestione overwrite immagini
    bool debug           = false;
    bool force_overwrite = false;

    // =======================
    // DNN - configurazione base
    // =======================
    // Tipo di rete (riconoscimento o detector, a seconda del profilo):
    //   caffe | tensorflow | onnx | openvino | tflite | torch
    std::string dnn_type        = "onnx";
    std::string dnn_model_path;   // modello generico (se non si usa profilo)
    std::string dnn_proto_path;   // config/prototxt, se applicabile
    std::string dnn_device   = "cpu"; // cpu|cuda|opencl|openvino
    double      dnn_threshold = 0.6;  // per embedding/detector DNN [0-1]

    // Profilo DNN logico (fast, sface, lresnet100, det_caffe, yunet, ...)
    std::string dnn_profile;

    // ==========================
    // DNN - modelli specifici
    // ==========================

    // 1) Reti di riconoscimento volto (embedding)
    std::string dnn_model_fast;
    std::string dnn_model_sface;
    std::string dnn_model_lresnet100;
    std::string dnn_model_openface;

    // 2) Detector volto (SSD / YuNet)
    std::string dnn_model_yunet;
    std::string dnn_model_detector_caffe;
    std::string dnn_model_detector_fp16;
    std::string dnn_model_detector_uint8;
    std::string dnn_proto_detector_caffe;

    // 3) Emotion / keypoints / MediaPipe TFLite
    std::string dnn_model_emotion;
    std::string dnn_model_keypoints;
    std::string dnn_model_face_landmark_tflite;
    std::string dnn_model_face_detection_tflite;
    std::string dnn_model_face_blendshapes_tflite;

    // ==========================
    // Haar cascade (fallback detector)
    // ==========================
    // Path assoluto del file haarcascade_frontalface_default.xml
    // Se vuoto, libfacialauth fa il fallback sui path di sistema noti.
    std::string haar_cascade;

    // Profilo logico per il DETECTOR (se vuoi distinguere dal profilo
    // usato per il riconoscimento). Può coincidere con dnn_profile
    // oppure essere qualcosa tipo "yunet", "det_caffe", ecc.
    std::string detector_profile;
};

// ============================================================
// API di configurazione
// ============================================================

// Carica i parametri da file INI-like.
// Ritorna true se il file esiste ed è stato letto con successo,
// false se il file non esiste o non è leggibile (in tal caso
// cfg rimane con i default).
bool fa_load_config(const std::string &path,
                    FacialAuthConfig  &cfg,
                    std::string       &log);

// Costruisce il path della directory immagini di un utente:
//   <cfg.basedir>/images/<user>/
std::string fa_user_image_dir(const FacialAuthConfig &cfg,
                              const std::string      &user);

// Costruisce il path del modello di un utente:
//   <cfg.basedir>/models/<user>.xml
std::string fa_user_model_path(const FacialAuthConfig &cfg,
                               const std::string      &user);

// Seleziona un profilo DNN logico (es. "sface", "fast", "yunet" ...)
// e imposta in cfg i campi dnn_type, dnn_model_path, dnn_proto_path, ecc.
bool fa_select_dnn_profile(FacialAuthConfig  &cfg,
                           const std::string &profile,
                           std::string       &log);

// ============================================================
// Wrapper di alto livello per riconoscitore + DNN
// ============================================================

class FaceRecWrapper {
public:
    FaceRecWrapper();
    explicit FaceRecWrapper(const std::string &modelType_);
    ~FaceRecWrapper() = default;

    // Configura se si usa un riconoscitore DNN (embedding) e
    // i relativi parametri dal FacialAuthConfig.
    void ConfigureDNN(const FacialAuthConfig &cfg);

    // Configura il detector (DNN YuNet / SSD o Haar) a partire dalla cfg.
    void ConfigureDetector(const FacialAuthConfig &cfg);

    // Carica/Salva modello (XML OpenCV + metadati custom fa_*)
    bool Load(const std::string &modelFile);
    bool Save(const std::string &modelFile) const;

    // Training:
    //  - metodi classici: addestra il FaceRecognizer
    //  - metodo "dnn": calcola un template di embedding medio
    bool Train(const std::vector<cv::Mat> &images,
               const std::vector<int>     &labels);

    // Predizione:
    //  - se in modalità DNN (IsDNN()), usa l’embedding+threshold
    //  - altrimenti usa il FaceRecognizer classico
    bool Predict(const cv::Mat &faceGray,
                 int           &label,
                 double        &confidence);

    // Individuazione volto nel frame:
    //  - se use_dnn_detector => DNN detector
    //  - altrimenti Haar cascade
    bool DetectFace(const cv::Mat &frame,
                    cv::Rect      &faceROI);

    // Info sullo stato DNN
    bool   IsDNN() const           { return use_dnn; }
    double GetDnnThreshold() const { return dnn_threshold; }

private:
    // Tipo di modello logico:
    //   "lbph" | "eigen" | "fisher" | "dnn"
    std::string modelType;

    // Riconoscitore classico OpenCV (usato sempre, anche come “dummy”
    // per produrre un XML valido nei modelli DNN)
    cv::Ptr<cv::face::FaceRecognizer> recognizer;

    // ==========================
    //  DNN di riconoscimento (embedding)
    // ==========================
    bool        use_dnn        = false; // true se vogliamo usare embedding DNN
    bool        dnn_loaded     = false; // true se rete caricata
    std::string dnn_profile;            // profilo logico
    std::string dnn_type;               // caffe|onnx|...
    std::string dnn_model_path;
    std::string dnn_proto_path;
    std::string dnn_device;             // cpu|cuda|...
    double      dnn_threshold  = 0.6;   // soglia logica [0-1]

    cv::dnn::Net dnn_net;               // rete per embedding
    cv::Mat      dnn_template;          // embedding medio (1 x N)
    bool         has_dnn_template = false;

    // ==========================
    //  Detector volto (DNN + Haar)
    // ==========================
    bool        use_dnn_detector     = false; // se true, prova prima DNN detector
    bool        dnn_detector_loaded  = false;
    std::string detector_profile;            // nome profilo detector (yunet, det_caffe, ...)
    std::string detector_type;               // caffe|onnx|...
    std::string detector_model_path;
    std::string detector_proto_path;
    std::string haar_cascade_path;          // path assoluto Haar, se configurato

    cv::dnn::Net        dnn_detector_net;   // rete per detezione volto
    cv::CascadeClassifier faceCascade;      // fallback Haar cascade

    // Helpers interni
    bool compute_dnn_embedding(const cv::Mat &faceGray,
                               cv::Mat       &embedding);

    bool predict_with_dnn(const cv::Mat &faceGray,
                          int           &label,
                          double        &confidence);

    // Ritorna true se il detector DNN “vede” un volto con score sufficiente.
    // (NOTA: non è const perché chiama setInput()/forward() sulla rete.)
    bool dnn_detector_accepts(const cv::Mat &frame);
};

// ============================================================
// API di alto livello per i tool (capture / train / test)
// ============================================================

// Cattura immagini per un utente partendo dalla cfg.
// - Se force/cfg.force_overwrite => cancella prima le immagini esistenti
// - img_format: "jpg", "png", ecc. (default "jpg")
bool fa_capture_images(const std::string      &user,
                       const FacialAuthConfig &cfg,
                       bool                    force,
                       std::string            &log,
                       const std::string      &img_format);

// Addestra un modello per l’utente:
//  - method: "lbph" | "eigen" | "fisher" | "dnn"
//  - inputDir: directory immagini (se vuoto => basedir/images/<user>/)
//  - outputModel: file XML (se vuoto => basedir/models/<user>.xml)
bool fa_train_user(const std::string      &user,
                   const FacialAuthConfig &cfg,
                   const std::string      &method,
                   const std::string      &inputDir,
                   const std::string      &outputModel,
                   bool                    force,
                   std::string            &log);

// Esegue un test in tempo reale dalla webcam usando il modello dell’utente.
// - modelPath: se vuoto => basedir/models/<user>.xml
// - best_conf/best_label: riportano il best score trovato
bool fa_test_user(const std::string      &user,
                  const FacialAuthConfig &cfg,
                  const std::string      &modelPath,
                  double                 &best_conf,
                  int                    &best_label,
                  std::string            &log);

// ============================================================
// Funzioni di manutenzione
// ============================================================

// Rimuove tutte le immagini di <user> (se la dir non esiste, ritorna true)
bool fa_clean_images(const FacialAuthConfig &cfg,
                     const std::string      &user);

// Rimuove il modello di <user> (se non esiste, ritorna true)
bool fa_clean_model(const FacialAuthConfig &cfg,
                    const std::string      &user);

// Stampa a stdout la lista delle immagini presenti per <user>
void fa_list_images(const FacialAuthConfig &cfg,
                    const std::string      &user);

// Verifica che il programma sia eseguito come root (geteuid()==0).
// In caso contrario stampa un errore e ritorna false.
bool fa_check_root(const char *tool_name = nullptr);

// ============================================================
// Helpers “pubblici” usati anche dai tool/CLI
// ============================================================

// Apre la camera in base alla cfg:
//  - Se cfg.device non è vuoto, apre quel device (es. /dev/video0).
//  - Altrimenti apre la camera cfg.camera_index.
// Imposta width/height se > 0.
// device_used viene valorizzato con il dispositivo effettivamente usato.
bool open_camera(const FacialAuthConfig &cfg,
                 cv::VideoCapture       &cap,
                 std::string            &device_used);

// Legge dinamicamente i profili DNN disponibili da un file di config,
// controllando quali dnn_model_* sono impostati.
// Restituisce un vettore ordinato e senza duplicati di nomi logici
// (es. "fast", "sface", "yunet", "det_caffe", ...).
std::vector<std::string>
fa_get_dnn_profiles_from_config(const std::string &config_path);

// ============================================================
// Entry-point CLI per i tre tool:
//   facial_training, facial_capture, facial_test
// ============================================================

// Implementazione principale di "facial_training".
int fa_training_cli(int argc, char *argv[]);

// Implementazione principale di "facial_capture".
int fa_capture_cli(int argc, char *argv[]);

// Implementazione principale di "facial_test".
int fa_test_cli(int argc, char *argv[]);

