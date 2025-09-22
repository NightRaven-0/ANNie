// training/train_ann.cpp
// Minimal, robust trainer for ANNie (5 inputs: front,left,right,diff,minLR)
// - Expects CSV: data/dataset.csv with header: front,left,right,diff,minLR,action
// - Trains a 5->32->16->4 MLP using tiny-dnn
// - Saves: models/ann_model_tinydnn.bin, models/predictions.csv, models/confusion.csv
//
// Compile (Developer Command Prompt):
// cl /EHsc /std:c++17 training\train_ann.cpp /I vendor\tiny-dnn /Fe:training\train_ann.exe
// Run:
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
  #error "C++17 <filesystem> required"
#endif

#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using std::string;
using std::vector;

struct Sample {
    std::array<float,5> x; // front,left,right,diff,minLR  (normalized)
    int y;                 // 0..3
};

static const float INPUT_RANGE_CM = 100.0f;

// Safe parse helpers
static inline bool str_to_float(const string &s, float &out) {
    try {
        size_t idx = 0;
        out = std::stof(s, &idx);
        return idx == s.size();
    } catch (...) { return false; }
}
static inline bool str_to_int(const string &s, int &out) {
    try {
        size_t idx = 0;
        out = std::stoi(s, &idx);
        return idx == s.size();
    } catch (...) { return false; }
}

// Load CSV with robust diagnostics
vector<Sample> load_dataset(const string &csv_path) {
    std::ifstream in(csv_path);
    if (!in.is_open()) {
        std::cerr << "ERROR: Cannot open dataset file: " << csv_path << std::endl;
        return {};
    }
    vector<Sample> out;
    string line;
    // Read header
    if (!std::getline(in, line)) {
        std::cerr << "ERROR: Dataset appears empty: " << csv_path << std::endl;
        return {};
    }
    std::cout << "CSV header: " << line << std::endl;

    size_t lineno = 1;
    size_t skipped = 0;
    while (std::getline(in, line)) {
        ++lineno;
        if (line.empty()) { ++skipped; continue; }
        std::stringstream ss(line);
        vector<string> cols;
        string tok;
        while (std::getline(ss, tok, ',')) cols.push_back(tok);

        if (cols.size() < 6) {
            std::cerr << "Skipping line " << lineno << " (expected 6 cols, got " << cols.size() << "): " << line << std::endl;
            ++skipped;
            continue;
        }

        float f,l,r,d,m;
        int a;
        bool ok = true;
        ok &= str_to_float(cols[0], f);
        ok &= str_to_float(cols[1], l);
        ok &= str_to_float(cols[2], r);
        ok &= str_to_float(cols[3], d);
        ok &= str_to_float(cols[4], m);
        ok &= str_to_int(cols[5], a);

        if (!ok) {
            std::cerr << "Skipping line " << lineno << " (parse error): " << line << std::endl;
            ++skipped;
            continue;
        }

        // Normalize / clamp
        float nf = std::clamp(f, 0.0f, INPUT_RANGE_CM) / INPUT_RANGE_CM;          // 0..1
        float nl = std::clamp(l, 0.0f, INPUT_RANGE_CM) / INPUT_RANGE_CM;          // 0..1
        float nr = std::clamp(r, 0.0f, INPUT_RANGE_CM) / INPUT_RANGE_CM;          // 0..1
        float nd = std::clamp(d, -INPUT_RANGE_CM, INPUT_RANGE_CM) / INPUT_RANGE_CM; // -1..1
        float nm = std::clamp(m, 0.0f, INPUT_RANGE_CM) / INPUT_RANGE_CM;          // 0..1

        Sample s;
        s.x = { nf, nl, nr, nd, nm };
        s.y = a;
        out.push_back(s);
    }

    std::cout << "Loaded " << out.size() << " valid samples, skipped " << skipped << " malformed/empty lines.\n";
    if (out.size() > 0) {
        std::cout << "Sample (first 5):\n";
        for (size_t i = 0; i < std::min<size_t>(5, out.size()); ++i) {
            auto &ss = out[i];
            std::cout << i << ": [";
            for (size_t j = 0; j < ss.x.size(); ++j) {
                std::cout << std::fixed << std::setprecision(3) << ss.x[j] << (j+1<ss.x.size()? ", ":"");
            }
            std::cout << "] -> " << ss.y << "\n";
        }
    }
    return out;
}

// Shuffle & split
void shuffle_split(const vector<Sample> &all, vector<Sample> &train, vector<Sample> &test, float test_ratio=0.2f, int seed=42) {
    vector<Sample> copy = all;
    std::mt19937 rng(seed);
    std::shuffle(copy.begin(), copy.end(), rng);
    size_t ntest = static_cast<size_t>(copy.size() * test_ratio);
    if (ntest > copy.size()) ntest = copy.size();
    test.assign(copy.begin(), copy.begin() + ntest);
    train.assign(copy.begin() + ntest, copy.end());
}

// Convert to tiny-dnn tensors
void to_tiny(const vector<Sample> &data, std::vector<vec_t> &X, std::vector<label_t> &Y) {
    X.clear(); Y.clear();
    X.reserve(data.size()); Y.reserve(data.size());
    for (const auto &s : data) {
        vec_t v;
        v.reserve(5);
        for (float val : s.x) v.push_back(val);
        X.push_back(v);
        Y.push_back(static_cast<label_t>(s.y));
    }
}

int main(int argc, char** argv) {
    try {
        // Paths
        fs::path repoRoot = fs::path(__FILE__).parent_path().parent_path();
        fs::path dataPath = repoRoot / "data" / "dataset.csv";
        fs::path modelsDir = repoRoot / "models";
        fs::create_directories(modelsDir);

        std::cout << "Loading dataset from: " << dataPath.string() << std::endl;
        auto all = load_dataset(dataPath.string());
        if (all.empty()) {
            std::cerr << "Dataset empty or not found. Make sure " << dataPath.string() << " exists and has header/front,left,right,diff,minLR,action\n";
            return 1;
        }

        // Print class distribution
        std::array<int,4> counts = {0,0,0,0};
        for (auto &s : all) if (s.y >=0 && s.y < 4) counts[s.y]++;
        std::cout << "Class counts: 0(FWD)=" << counts[0] << " 1(LEFT)=" << counts[1] << " 2(RIGHT)=" << counts[2] << " 3(STOP)=" << counts[3] << "\n";

        // Split
        vector<Sample> train_samples, test_samples;
        shuffle_split(all, train_samples, test_samples, 0.2f, 1234);
        std::cout << "Train: " << train_samples.size() << "  Test: " << test_samples.size() << std::endl;

        // Convert to tiny-dnn
        std::vector<vec_t> X_train, X_test;
        std::vector<label_t> y_train, y_test;
        to_tiny(train_samples, X_train, y_train);
        to_tiny(test_samples, X_test, y_test);

        // Build network: 5 -> 32 -> 16 -> 4
        network<sequential> net;
        net << fully_connected_layer(5, 64) << relu()
            << fully_connected_layer(64, 32) << relu()
            << fully_connected_layer(32, 16) << relu()
            << fully_connected_layer(16, 4);

        // Optimizer
        adam optimizer;
        optimizer.alpha = float_t(1e-3);

        const int epochs = 300;
        const int batch_size = 32;

        std::cout << "Starting training (epochs=" << epochs << ", batch=" << batch_size << ")...\n";
        progress_display disp(static_cast<unsigned long>(X_train.size()));
        timer t;
        // tiny-dnn offers callbacks for progress_display; use simple train call
        net.train<cross_entropy_multiclass>(optimizer, X_train, y_train, batch_size, epochs);
        std::cout << "Training complete.\n";

        // Evaluate
        int correct = 0;
        const int nClasses = 4;
        std::vector<std::vector<int>> confusion(nClasses, std::vector<int>(nClasses,0));
        std::ofstream predout((modelsDir / "predictions.csv").string());
        predout << "f,l,r,diff,minLR,label,pred\n";
        for (size_t i = 0; i < X_test.size(); ++i) {
            vec_t res = net.predict(X_test[i]);
            int pred = static_cast<int>(std::distance(res.begin(), std::max_element(res.begin(), res.end())));
            int truth = static_cast<int>(y_test[i]);
            if (pred == truth) ++correct;
            if (truth >=0 && truth < nClasses) confusion[truth][pred]++;
            // write original normalized floats (for inspection)
            predout << std::fixed << std::setprecision(5)
                    << X_test[i][0] << "," << X_test[i][1] << "," << X_test[i][2] << ","
                    << X_test[i][3] << "," << X_test[i][4] << "," << truth << "," << pred << "\n";
        }
        predout.close();

        float acc = (X_test.empty() ? 0.0f : float(correct) / float(X_test.size()));
        std::cout << "Test accuracy: " << acc << " (" << correct << "/" << X_test.size() << ")\n";

        // Save confusion
        std::ofstream cfout((modelsDir / "confusion.csv").string());
        cfout << "label/pred,0,1,2,3\n";
        for (int i = 0; i < nClasses; ++i) {
            cfout << i;
            for (int j = 0; j < nClasses; ++j) cfout << "," << confusion[i][j];
            cfout << "\n";
        }
        cfout.close();
        std::cout << "Saved predictions.csv and confusion.csv to " << modelsDir.string() << "\n";

        // Save model binary
        std::string modelBinPath = (modelsDir / "ann_model_tinydnn.bin").string();
        net.save(modelBinPath);
        std::cout << "Saved tiny-dnn model to: " << modelBinPath << "\n";

        std::cout << "Done.\n";
        return 0;
    } catch (const std::exception &ex) {
        std::cerr << "EXCEPTION: " << ex.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "EXCEPTION: unknown\n";
        return 1;
    }
}
