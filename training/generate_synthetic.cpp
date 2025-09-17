// ./training/gen_data
//compile with :
// cl /EHsc /std:c++17 training\generate_synthetic.cpp /Fe:training\gen_data.exe
//run with:
// training\gen_data.exe

#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <filesystem>
using namespace std;

int main() {
    std::filesystem::create_directories("data");
    ofstream fout("data/dataset.csv");
    fout << "front,left,right,action\n";

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> far(40.0f, 120.0f);
    std::uniform_real_distribution<float> mid(15.0f, 60.0f);
    std::uniform_real_distribution<float> near(0.0f, 25.0f);
    std::bernoulli_distribution coin(0.5);

    // Generate scenarios
    for (int i = 0; i < 200; ++i) {
        // mostly clear forward
        float f = far(rng);
        float l = mid(rng);
        float r = mid(rng);
        int action = 0; // FORWARD
        // add some random closer obstacles
        if (coin(rng)) { f = mid(rng); }
        if (coin(rng)) { l = near(rng); }
        if (coin(rng)) { r = near(rng); }
        // label heuristics: if front far -> forward, else turn to side with more space, else stop
        if (f > 35.0f) action = 0;
        else if (l > r && l > 20.0f) action = 1; // LEFT
        else if (r > l && r > 20.0f) action = 2; // RIGHT
        else action = 3; // STOP
        fout << int(f+0.5f) << "," << int(l+0.5f) << "," << int(r+0.5f) << "," << action << "\n";
    }

    // generate close-obstacle heavy samples
    for (int i = 0; i < 200; ++i) {
        float f = near(rng);
        float l = near(rng);
        float r = near(rng);
        int action;
        if (l > r && l > 15.0f) action = 1;
        else if (r > l && r > 15.0f) action = 2;
        else if (f > 20.0f) action = 0;
        else action = 3;
        fout << int(f+0.5f) << "," << int(l+0.5f) << "," << int(r+0.5f) << "," << action << "\n";
    }

    fout.close();
    cout << "Wrote data/dataset.csv with 400 rows (approx)\n";
    return 0;
}
