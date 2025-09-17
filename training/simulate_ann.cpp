// simulate_ann.cpp
// Purpose: Load trained tiny-dnn model and simulate ANNie decisions on sample inputs
// Build: cl /EHsc /std:c++17 training\simulate_ann.cpp /I vendor\tiny-dnn /Fe:training\simulate_ann.exe
// run: training\simulate_ann.exe

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn;

int main() {
    // Load trained model
    network<sequential> net;
    try {
        net.load("models/ann_model_tinydnn.bin");
    } catch (const std::exception &e) {
        std::cerr << "Failed to load model: " << e.what() << "\n";
        return 1;
    }
    std::cout << "Loaded trained model from models/ann_model_tinydnn.bin\n";

    // Define test scenarios
    std::vector<vec_t> demo_inputs = {
        {0.9f, 0.5f, 0.5f}, // clear forward
        {0.1f, 0.8f, 0.2f}, // blocked front, open left
        {0.2f, 0.2f, 0.9f}, // blocked front, open right
        {0.1f, 0.1f, 0.1f}, // blocked all sides
        {0.5f, 0.9f, 0.9f}, // mid forward, open sides
    };

    std::vector<std::string> labels = {"FORWARD", "LEFT", "RIGHT", "STOP"};

    // Print predictions
    for (size_t i = 0; i < demo_inputs.size(); i++) {
        auto res = net.predict(demo_inputs[i]);
        int pred = std::distance(res.begin(), std::max_element(res.begin(), res.end()));
        std::cout << "Case " << i 
                  << " input(" << demo_inputs[i][0] << "," << demo_inputs[i][1] << "," << demo_inputs[i][2] 
                  << ") -> " << labels[pred] << "\n";
    }

    // Export results to CSV
    std::ofstream fout("results.csv");
    fout << "front,left,right,predicted\n";
    for (size_t i = 0; i < demo_inputs.size(); i++) {
        auto res = net.predict(demo_inputs[i]);
        int pred = std::distance(res.begin(), std::max_element(res.begin(), res.end()));
        fout << demo_inputs[i][0] << "," << demo_inputs[i][1] << "," << demo_inputs[i][2] << "," << labels[pred] << "\n";
    }
    fout.close();
    std::cout << "Saved results to results.csv\n";

    return 0;
}
