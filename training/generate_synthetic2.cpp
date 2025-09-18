// generate_synthetic2.cpp
// Improved synthetic dataset generator for ANNie
// Produces balanced, varied data and writes data/dataset.csv
//
// Build (MSVC Developer Prompt):
// cl /EHsc /std:c++17 training\generate_synthetic2.cpp /Fe:training\gen_data2.exe
// Run:
// training\gen_data2.exe

#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <string>
#include <array>        // <<-- important

int main() {
    std::filesystem::create_directories("data");
    std::ofstream fout("data/dataset.csv");
    if (!fout.is_open()) {
        std::cerr << "Failed to open data/dataset.csv for writing\n";
        return 1;
    }

    fout << "front,left,right,action\n";

    const int TOTAL = 3000;           // total rows to generate
    const int CLASSES = 4;            // 0=FORWARD,1=LEFT,2=RIGHT,3=STOP
    const int PER_CLASS = TOTAL / CLASSES; // aim for balanced

    std::mt19937 rng(12345);
    std::normal_distribution<float> noise(0.0f, 2.5f);   // small gaussian noise (cm)
    std::uniform_real_distribution<float> uniform_far(60.0f, 120.0f);
    std::uniform_real_distribution<float> uniform_mid(20.0f, 60.0f);
    std::uniform_real_distribution<float> uniform_near(0.0f, 25.0f);
    std::bernoulli_distribution coin(0.5);
    std::uniform_real_distribution<float> prob(0.0f, 1.0f);

    auto clip = [](float v, float lo, float hi) {
        if (v < lo) return lo;
        if (v > hi) return hi;
        return v;
    };

    std::vector<std::array<float,4>> rows; // front,left,right,action

    // Helper to push row with rounding
    auto push_row = [&](float f, float l, float r, int a){
        // add sensor noise
        f = clip(f + noise(rng), 0.0f, 9999.0f);
        l = clip(l + noise(rng), 0.0f, 9999.0f);
        r = clip(r + noise(rng), 0.0f, 9999.0f);

        // occasionally make sensor out-of-range (simulate no echo)
        if (prob(rng) < 0.01) f = 999.0f;
        if (prob(rng) < 0.01) l = 999.0f;
        if (prob(rng) < 0.01) r = 999.0f;

        std::array<float,4> arr = { f, l, r, static_cast<float>(a) }; // construct then push
        rows.push_back(arr);
    };

    // 1) FORWARD examples: front clear, sides varied
    for (int i = 0; i < PER_CLASS; ++i) {
        float f = uniform_far(rng);
        float l = uniform_mid(rng);
        float r = uniform_mid(rng);
        // Add some borderline front cases occasionally
        if (i % 10 == 0) f = 30.0f + std::abs(noise(rng))*2.0f;
        push_row(f, l, r, 0);
    }

    // 2) LEFT examples: left clear, right or front blocked
    for (int i = 0; i < PER_CLASS; ++i) {
        float l = uniform_far(rng);
        float r = uniform_near(rng);
        float f = (coin(rng) ? uniform_mid(rng) : uniform_near(rng));
        if (i % 7 == 0) r = 15.0f + (uniform_near(rng));
        push_row(f, l, r, 1);
    }

    // 3) RIGHT examples: right clear, left or front blocked
    for (int i = 0; i < PER_CLASS; ++i) {
        float r = uniform_far(rng);
        float l = uniform_near(rng);
        float f = (coin(rng) ? uniform_mid(rng) : uniform_near(rng));
        if (i % 7 == 0) l = 15.0f + (uniform_near(rng));
        push_row(f, l, r, 2);
    }

    // 4) STOP examples: front blocked and both sides poor (close)
    for (int i = 0; i < PER_CLASS; ++i) {
        float f = uniform_near(rng);
        float l = uniform_near(rng);
        float r = uniform_near(rng);
        // some STOP cases where one side slightly better but still low
        if (i % 6 == 0) {
            if (coin(rng)) l = 30.0f + uniform_near(rng);
            else r = 30.0f + uniform_near(rng);
        }
        push_row(f, l, r, 3);
    }

    // 5) Add focused edge cases / borderline mix (rest to reach TOTAL)
    int generated = static_cast<int>(rows.size());
    while (generated < TOTAL) {
        float choice = prob(rng);
        if (choice < 0.25f) {
            // borderline front with one side slightly better
            float f = 22.0f + uniform_near(rng);
            float l = 25.0f + uniform_mid(rng);
            float r = 15.0f + uniform_near(rng);
            int label = (l > r ? 1 : 2);
            if (f > 30.0f) label = 0;
            push_row(f, l, r, label);
        } else if (choice < 0.5f) {
            // shallow obstacle low enough to pass (simulate higher clearance)
            float f = 18.0f + uniform_near(rng);
            float l = 80.0f + uniform_mid(rng);
            float r = 80.0f + uniform_mid(rng);
            push_row(f, l, r, 0);
        } else if (choice < 0.75f) {
            // random mix
            float f = uniform_mid(rng);
            float l = uniform_mid(rng);
            float r = uniform_mid(rng);
            int label = (f > 40.0f) ? 0 : (l > r ? 1 : 2);
            push_row(f, l, r, label);
        } else {
            // simulate sensor dropout front (999), choose side
            float f = 999.0f;
            float l = uniform_far(rng);
            float r = uniform_near(rng);
            int label = (l > r ? 1 : 2);
            push_row(f, l, r, label);
        }
        generated = static_cast<int>(rows.size());
    }

    // Shuffle rows to mix classes
    std::shuffle(rows.begin(), rows.end(), rng);

    // Write to CSV (rounded ints for clarity)
    for (const auto &row : rows) {
        int f = (row[0] >= 999.0f ? 999 : static_cast<int>(std::round(row[0])));
        int l = (row[1] >= 999.0f ? 999 : static_cast<int>(std::round(row[1])));
        int r = (row[2] >= 999.0f ? 999 : static_cast<int>(std::round(row[2])));
        int a = static_cast<int>(std::round(row[3]));
        fout << f << "," << l << "," << r << "," << a << "\n";
    }

    fout.close();
    std::cout << "Wrote data/dataset.csv with " << rows.size() << " rows\n";

    // Print class distribution (quick)
    std::vector<int> counts(4,0);
    for (const auto &row : rows) counts[static_cast<int>(row[3])]++;
    std::cout << "Class counts: FORWARD=" << counts[0] << " LEFT=" << counts[1]
              << " RIGHT=" << counts[2] << " STOP=" << counts[3] << "\n";

    return 0;
}