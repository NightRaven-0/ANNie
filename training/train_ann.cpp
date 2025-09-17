// Purpose:
//  - Load ANNie/data/dataset.csv with columns: front,left,right,action
//  - Train a small MLP (3 -> 16 -> 8 -> 4) using tiny-dnn
//  - Save model weights as ANNie/models/arduino_weights.h (C arrays of floats)
//  - Save a tiny-dnn binary model file ANNie/models/ann_model_tinydnn.bin (optional)
// Notes:
//  - The CSV must have a header line: front,left,right,action
//  - action values: 0=FORWARD,1=LEFT,2=RIGHT,3=STOP
//  - Normalization: distances clipped to [0,100] cm then scaled to [0..1]
//  - tiny-dnn: Put tiny-dnn in vendor/tiny-dnn, include with -Ivendor/tiny-dnn
// compile with 
// cl /EHsc /std:c++17 training\train_ann.cpp /I vendor\tiny-dnn /Fe:training\train_ann.exe
// run with 
// training\train_ann.exe

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <iomanip>
#include <array>
#include <stdexcept>

#if __has_include(<filesystem>)
  #include <filesystem>
  namespace fs = std::filesystem;
#else
  // If your compiler doesn't have <filesystem>, you can fallback to C-style path handling.
  // But most modern compilers with -std=c++17 have filesystem.
  #error "C++17 filesystem required"
#endif

// tiny-dnn include (adjust include path on compile with -I)
#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using std::string;
using std::vector;

struct Sample 
{
    std::array<float,3> x;
    int y; // 0..3
};

static const float INPUT_RANGE_CM = 100.0f;

// Utility: read CSV dataset
vector<Sample> load_dataset(const string &csv_path) 
{
    std::ifstream in(csv_path);
    if (!in.is_open()) 
    {
        throw std::runtime_error("Failed to open dataset: " + csv_path);
    }
    vector<Sample> out;
    string line;
    // read header
    if (!std::getline(in, line)) throw std::runtime_error("Empty CSV or missing header");
    while (std::getline(in, line)) 
    {
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
void shuffle_split(const vector<Sample> &all, vector<Sample> &train, vector<Sample> &test, float test_ratio=0.2f, int seed=42) 
{
    vector<Sample> copy = all;
    std::mt19937 rng(seed);
    std::shuffle(copy.begin(), copy.end(), rng);
    size_t ntest = static_cast<size_t>(copy.size() * test_ratio);
    if (ntest > copy.size()) ntest = copy.size();
    test.assign(copy.begin(), copy.begin() + ntest);
    train.assign(copy.begin() + ntest, copy.end());
}

// Convert to tiny-dnn tensors
void to_tiny(const vector<Sample> &data, std::vector<vec_t> &X, std::vector<label_t> &Y) 
{
    X.clear(); Y.clear();
    X.reserve(data.size());
    Y.reserve(data.size());
    for (size_t i = 0; i < data.size(); ++i) 
    {
        vec_t v;
        v.push_back(data[i].x[0]);
        v.push_back(data[i].x[1]);
        v.push_back(data[i].x[2]);
        X.push_back(v);
        Y.push_back(static_cast<label_t>(data[i].y));
    }
}

// Export weights & biases to a C header usable by Arduino
// We use index-based access to avoid range-for issues on different tiny-dnn forks.
void export_weights_header(network<sequential> &net, const std::string &out_path) 
{
    std::ofstream fh(out_path);
    if (!fh.is_open()) throw std::runtime_error("Cannot open weights header for writing");
    fh << "// Auto-generated weights header from train_ann.cpp\n";
    fh << "#ifndef ANN_WEIGHTS_H\n#define ANN_WEIGHTS_H\n\n";
    for (size_t li = 0; li < net.depth(); ++li) 
    {
        auto layer_ptr = net[li];
        // In your fork: weights() returns vector<vec_t*>
        auto params = layer_ptr->weights();
        if (params.empty()) continue;
        for (size_t pidx = 0; pidx < params.size(); ++pidx) 
        {
            vec_t* vec_ptr = params[pidx];
            if (!vec_ptr) continue;
            const vec_t &vec = *vec_ptr;
            std::ostringstream varname;
            varname << "L" << li << "_P" << pidx;
            fh << "// Layer " << li << " param " << pidx
               << " length=" << vec.size() << "\n";
            fh << "const float " << varname.str() << "[] = { \n";
            for (size_t k = 0; k < vec.size(); ++k) {
                fh << std::fixed << std::setprecision(8) << vec[k] << "f";
                if (k + 1 < vec.size()) fh << ", ";
            }
            fh << "\n};\n\n";
        }
    }
    fh << "#endif // ANN_WEIGHTS_H\n";
    fh.close();
    std::cout << "Wrote weights header: " << out_path << std::endl;
}

int main(int argc, char** argv) 
{
    try 
    {
        // Build paths relative to this file's folder (training/)
        fs::path repoRoot = fs::path(__FILE__).parent_path().parent_path(); // up from training/
        fs::path dataPath = repoRoot / "data" / "dataset.csv";
        fs::path modelsDir = repoRoot / "models";
        fs::create_directories(modelsDir);
        std::cout << "Loading dataset from: " << dataPath << std::endl;
        auto all = load_dataset(dataPath.string());
        if (all.empty()) 
        {
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
        // Build network: 3 -> 16 (relu) -> 8 (relu) -> 4 (logits)
        network<sequential> net;
        net << fully_connected_layer(3, 16) << relu()
            << fully_connected_layer(16, 8) << relu()
            << fully_connected_layer(8, 4);
        // Optimizer
        adagrad optimizer;
        optimizer.alpha = float_t(0.01);
        // Train parameters
        const int epochs = 80;
        const int batch_size = 32;
        std::cout << "Starting training...\n";
        // train: tiny-dnn will handle shuffling if requested; we use train() convenience
        net.train<cross_entropy_multiclass>(optimizer, X_train, y_train, batch_size, epochs);
        std::cout << "Training complete.\n";
        // Evaluate accuracy on test set
        int correct = 0;
        for (size_t i = 0; i < X_test.size(); ++i) 
        {
            vec_t res = net.predict(X_test[i]);
            int pred = static_cast<int>(std::distance(res.begin(), std::max_element(res.begin(), res.end())));
            if (pred == y_test[i]) ++correct;
        }
        float acc = (X_test.empty() ? 0.0f : float(correct) / float(X_test.size()));
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
    } 
    catch (const std::exception &ex) 
    {
        std::cerr << "Exception: " << ex.what() << std::endl;
        return 1;
    }
}