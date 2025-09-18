// generate_synthetic3.cpp
// Generates clean, strictly-labeled dataset for ANN obstacle avoidance
//
// Build (MSVC Developer Prompt):
// cl /EHsc /std:c++17 training\generate_synthetic3.cpp /Fe:training\gen_data3.exe
// Run:
// training\gen_data3.exe

#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <algorithm>
#include <cmath>
#include <filesystem>

int main() {
    std::filesystem::create_directories("data");
    std::ofstream fout("data/dataset.csv");
    if (!fout.is_open()) {
        std::cerr << "Failed to open data/dataset.csv for writing\n";
        return 1;
    }

    fout << "front,left,right,action\n";

    const int TARGET_PER_CLASS = 1000; // generate ~1000 examples per class
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 100.0f);

    int count[4] = {0,0,0,0};
    int max_total = TARGET_PER_CLASS * 4;

    while (count[0] < TARGET_PER_CLASS || count[1] < TARGET_PER_CLASS ||
           count[2] < TARGET_PER_CLASS || count[3] < TARGET_PER_CLASS) {

        float f = dist(rng);
        float l = dist(rng);
        float r = dist(rng);
        int label = -1;

        // Apply strict rules
        if (f > 70 && l > 30 && r > 30) {
            label = 0; // FORWARD
        } else if (f < 40 && l > 50 && r < 30) {
            label = 1; // LEFT
        } else if (f < 40 && r > 50 && l < 30) {
            label = 2; // RIGHT
        } else if (f < 20 && l < 20 && r < 20) {
            label = 3; // STOP
        }

        if (label != -1 && count[label] < TARGET_PER_CLASS) {
            fout << int(f) << "," << int(l) << "," << int(r) << "," << label << "\n";
            count[label]++;
        }

        // Stop if we've reached total target
        int total = count[0] + count[1] + count[2] + count[3];
        if (total >= max_total) break;
    }

    fout.close();

    std::cout << "Wrote data/dataset.csv\n";
    std::cout << "Counts: FORWARD=" << count[0]
              << " LEFT=" << count[1]
              << " RIGHT=" << count[2]
              << " STOP=" << count[3] << "\n";
    return 0;
}