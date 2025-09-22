// training/generate_synthetic.cpp
// Generate synthetic dataset for ANNie with clear, rule-based labels
// Each row: front,left,right,diff,minLR,action
// Compile: cl /EHsc /std:c++17 training\generate_synthetic.cpp /Fe:training\gen_data.exe
// Run: training\gen_data.exe

#include <iostream>
#include <fstream>
#include <random>
#include <filesystem>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    try {
        fs::path repoRoot = fs::path(__FILE__).parent_path().parent_path();
        fs::path dataPath = repoRoot / "data" / "dataset.csv";
        fs::create_directories(repoRoot / "data");

        std::ofstream out(dataPath);
        if (!out.is_open()) {
            std::cerr << "Failed to open output file: " << dataPath << std::endl;
            return 1;
        }

        const int N = 12000; // dataset size
        out << "front,left,right,diff,minLR,action\n";

        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(0.0f, 100.0f);

        int countF=0, countL=0, countR=0, countS=0;

        for (int i = 0; i < N; i++) {
            float front = dist(rng);
            float left  = dist(rng);
            float right = dist(rng);
            float diff  = left - right;
            float minLR = (left < right ? left : right);

            int action;
            if (front > 70.0f) {
                action = 0; // FORWARD
                countF++;
            } else if (front <= 70.0f && (left - right) > 5.0f) {
                action = 1; // LEFT
                countL++;
            } else if (front <= 70.0f && (right - left) > 5.0f) {
                action = 2; // RIGHT
                countR++;
            } else {
                action = 3; // STOP (front blocked & sides nearly equal)
                countS++;
            }

            out << front << "," << left << "," << right << ","
                << diff << "," << minLR << "," << action << "\n";
        }

        out.close();

        std::cout << "Wrote dataset: " << dataPath << " with " << N << " rows.\n";
        std::cout << "Class counts -> "
                  << "FORWARD=" << countF << ", "
                  << "LEFT=" << countL << ", "
                  << "RIGHT=" << countR << ", "
                  << "STOP=" << countS << std::endl;

        return 0;
    } catch (const std::exception &ex) {
        std::cerr << "Exception: " << ex.what() << std::endl;
        return 1;
    }
}
