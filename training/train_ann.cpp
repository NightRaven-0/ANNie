// path to tiny-dnn P:\AML\tiny-dnn

// Purpose:
//  - Load ANNie/data/dataset.csv with columns: front,left,right,action
//  - Train a small MLP (3 -> 16 -> 8 -> 4) using tiny-dnn
//  - Save model weights as ANNie/models/arduino_weights.h (C arrays of floats)
//  - Save a tiny-dnn binary model file ANNie/models/ann_model_tinydnn.bin (optional)
//
// Notes:
//  - The CSV must have a header line: front,left,right,action
//  - action values: 0=FORWARD,1=LEFT,2=RIGHT,3=STOP
//  - Normalization: distances clipped to [0,100] cm then scaled to [0..1]
//  - tiny-dnn: https://github.com/tiny-dnn/tiny-dnn  (header-only)

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <iomanip>
#include <filesystem>

#include "tiny_dnn/tiny_dnn.h" // adjust include path to your tiny-dnn include

namespace fs = std::filesystem;
using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using std::string;
using std::vector;

struct Sample {
    std::array<float,3> x;
    int y; // 0..3
};

static const float INPUT_RANGE_CM = 100.0f;
static const std::string ROOT = ".."; // run from ANNie/training/ ; adjust if running elsewhere

// Utility: read CSV dataset
vector<Sample> load_dataset(const string &csv_path) {
    std::ifstream in(csv_path);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open dataset: " + csv_path);
    }
    vector<Sample> out;
    string line;
    // read header
    if (!std::getline(in, line)) throw std::runtime_error("Empty CSV");
    while (std::getline(in, line)) {
        if (line.size() < 3) continue;
        std::stringstream ss(line);
        string tok;
        vector<string> cols;
        while (std::getline(ss, tok, ',')) cols.push_back(tok);
        if (cols.size() < 4) continue;
        float f = std::stof(cols[0]);
        float l = std::stof(cols[1]);
        float r = std::stof(cols[2]);
        int a = std::stoi(cols[3]);
        // clip
        f = std::clamp(f, 0.0f, INPUT_RANGE_CM);
        l = std::clamp(l, 0.0f, INPUT_RANGE_CM);
        r = std::clamp(r, 0.0f, INPUT_RANGE_CM);
        Sample s;
        s.x = { f / INPUT_RANGE_CM, l / INPUT_RANGE_CM, r / INPUT_RANGE_CM }; // normalize 0..1
        s.y = a;
        out.push_back(s);
    }
    return out;
}

// Shuffle & split
void shuffle_split(const vector<Sample> &all, vector<Sample> &train, vector<Sample> &test, float test_ratio=0.2f, int seed=42) {
    vector<Sample> copy = all;
    std::mt19937 rng(seed);
    std::shuffle(copy.begin(), copy.end(), rng);
    size_t ntest = static_cast<size_t>(copy.size() * test_ratio);
    test.assign(copy.begin(), copy.begin() + ntest);
    train.assign(copy.begin() + ntest, copy.end());
}

// Convert to tiny-dnn tensors
void to_tiny(const vector<Sample> &data, std::vector<vec_t> &X, std::vector<label_t> &Y) {
    X.clear(); Y.clear();
    for (const auto &s : data) {
        vec_t v;
        v.push_back(s.x[0]);
        v.push_back(s.x[1]);
        v.push_back(s.x[2]);
        X.push_back(v);
        Y.push_back(static_cast<label_t>(s.y));
    }
}

// Export weights & biases to a C header usable by Arduino
// We'll extract params from each layer (dense layers) and write flat arrays.
// Note: tiny-dnn stores weights in layer->weight_ptr() etc.
void export_weights_header(network<sequential> &net, const string &out_path) {
    std::ofstream fh(out_path);
    if (!fh.is_open()) throw std::runtime_error("Cannot open weights header for writing");

    fh << "// Auto-generated weights header from train_ann.cpp\n";
    fh << "#ifndef ANN_WEIGHTS_H\n#define ANN_WEIGHTS_H\n\n";

    int layer_index = 0;
    for (auto &layer_ptr : net) {
        // We only export fully-connected (fc) layers with parameters
        auto params = layer_ptr->weights();
        if (params.empty()) { ++layer_index; continue; }
        // tiny-dnn weights() returns vector<tensor_t> where first is W, second is b typically
        // We'll flatten each param tensor
        for (size_t pidx = 0; pidx < params.size(); ++pidx) {
            // params[pidx] is tensor_t -> vector<vec_t> where each vec_t is a column/row chunk
            // We'll flatten by concatenating all floats.
            std::ostringstream varname;
            varname << "L" << layer_index << "_P" << pidx;
            // Determine total length
            size_t total = 0;
            for (auto &tt : params[pidx]) total += tt.size();

            fh << "// Layer " << layer_index << " param " << pidx << " shape flattened, total=" << total << "\n";
            fh << "const float " << varname.str() << "[] = { \n";
            bool first = true;
            for (const auto &tt : params[pidx]) {
                for (float v : tt) {
                    fh << std::fixed << std::setprecision(8) << v << "f, ";
                }
            }
            fh << "\n};\n\n";
        }
        ++layer_index;
    }

    fh << "#endif // ANN_WEIGHTS_H\n";
    fh.close();
    std::cout << "Wrote weights header: " << out_path << std::endl;
}

int main(int argc, char** argv) {
    try {
        // Paths
        fs::path repoRoot = fs::path(_FILE_).parent_path().parent_path(); // go up from training/
        fs::path dataPath = repoRoot / "data" / "dataset.csv";
        fs::path modelsDir = repoRoot / "models";
        fs::create_directories(modelsDir);

        std::cout << "Loading dataset from: " << dataPath << std::endl;
        auto all = load_dataset(dataPath.string());
        if (all.empty()) {
            std::cerr << "Dataset empty or not found. Create data/dataset.csv with front,left,right,action\n";
            return 1;
        }
        std::cout << "Loaded " << all.size() << " samples\n";

        // Split
        vector<Sample> train_samples, test_samples;
        shuffle_split(all, train_samples, test_samples, 0.2f, 42);
        std::cout << "Train: " << train_samples.size() << " Test: " << test_samples.size() << std::endl;

        // Convert to tiny-dnn data structures
        std::vector<vec_t> X_train, X_test;
        std::vector<label_t> y_train, y_test;
        to_tiny(train_samples, X_train, y_train);
        to_tiny(test_samples, X_test, y_test);

        // Build network: 3 -> 16 (relu) -> 8 (relu) -> 4 (softmax via cross-entropy)
        network<sequential> net;
        net << fully_connected_layer(3, 16) << relu()
            << fully_connected_layer(16, 8) << relu()
            << fully_connected_layer(8, 4); // logits, loss function will apply softmax

        // Optimizer
        adagrad optimizer; // simple optimizer; alternatives: adam, momentum
        optimizer.alpha = float_t(0.01);

        // Train parameters
        const int epochs = 80;
        const int batch_size = 32;

        progress_display disp(X_train.size());
        timer t;

        std::cout << "Starting training...\n";
        // tiny-dnn train() accepts vectors; cross_entropy_multiclass is the loss
        net.train<cross_entropy_multiclass>(optimizer, X_train, y_train, batch_size, epochs,
            [&](){ /* epoch begin */ },
            [&](){ /* epoch end */ }
        );

        std::cout << "Training complete.\n";

        // Evaluate accuracy on test set
        int correct = 0;
        for (size_t i = 0; i < X_test.size(); ++i) {
            auto res = net.predict(X_test[i]);
            // res is vec_t of length 4 logits; find argmax
            int pred = std::distance(res.begin(), std::max_element(res.begin(), res.end()));
            if (pred == y_test[i]) ++correct;
        }
        float acc = float(correct) / float(X_test.size());
        std::cout << "Test accuracy: " << acc << " (" << correct << "/" << X_test.size() << ")\n";

        // Save the tiny-dnn model (binary)
        auto modelBinPath = (modelsDir / "ann_model_tinydnn.bin").string();
        net.save(modelBinPath);
        std::cout << "Saved tiny-dnn model to: " << modelBinPath << "\n";

        // Export weights header suitable for inclusion in Arduino firmware
        auto headerPath = (modelsDir / "arduino_weights.h").string();
        export_weights_header(net, headerPath);

        std::cout << "All done. Outputs in models/ (arduino_weights.h and model binary)\n";
        return 0;
    } catch (const std::exception &ex) {
        std::cerr << "Exception: " << ex.what() << std::endl;
        return 1;
    }
}