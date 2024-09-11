#include <iostream>
#include <MLP.cpp>
#include "print_vec.h"
#include <random>

using namespace std;

float generateRandomFloat(float min, float max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);
    return dis(gen);
}
vector<vector<vector<double>>> dataGenerator(int s) {
    vector<vector<double>>inputs;
    vector<vector<double>> outputs;
    for (int i = 0; i < s; ++i) {
        double x = generateRandomFloat(0,1);
        double y = x * x;
        inputs.push_back({ x });
        outputs.push_back({ y });
    }
    return
     { inputs,outputs };
}
int main()
{
    try
    {
        int total_presentations = 100;
        vector<vector<vector<double>>>vec= dataGenerator(total_presentations);
        vector<vector<double>> inputs = vec[0];
        vector<vector<double>> desired =vec[1];

        vector<vector<vector<double>>> weights = {
            {{1}},
            {{generateRandomFloat(-1, 1)},{generateRandomFloat(-1, 1)}, {generateRandomFloat(-1, 1)}},
            {{generateRandomFloat(-1, 1), generateRandomFloat(-1, 1), generateRandomFloat(-1, 1)}}
        };

        vector<vector<double>> bias = {
            {0},
            {generateRandomFloat(-1, 1), generateRandomFloat(-1, 1), generateRandomFloat(-1, 1)},
            {generateRandomFloat(-1, 1)}
        };

        vector<int> nodes_per_layer = { 1, 3, 1 };


        vector<double> input_test = { 0.23 };
        MLP mlp(3, nodes_per_layer);
        mlp.initialize_neurons(weights, bias);
        mlp.train(total_presentations, inputs, desired);
        while (true) {
            double inp;
            cin >> inp;

            cout << mlp.predict({ inp}) << endl;
        }

}
    
    catch (exception e)
    {
        cout << e.what();
    }
    return 0;
}
