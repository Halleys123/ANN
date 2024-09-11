#pragma once

#include "data_generator.h"
#include "print_vec.h"
#include "MLP.h"

#include <iostream>
#include <random>

using namespace std;

int main()
{
    try
    {
        int total_presentations = 100;

        vector<vector<vector<double>>>vec = dataGenerator(total_presentations);
        vector<vector<double>> inputs = vec[0];
        vector<vector<double>> desired =vec[1];
        
        //vector<vector<double>> inputs = { {0.1}, {0.2}, {0.5} };
        //vector<vector<double>> desired = { {0.01}, {0.02}, {0.25} };

        vector<int> nodes_per_layer = { 1, 4,  1 };

        MLP mlp(nodes_per_layer.size(), nodes_per_layer);

        mlp.set_learning_const(0.5);
        mlp.set_error_limit(0.01);
        mlp.train(total_presentations, inputs, desired);

        while (true) {
            cout << "Enter a number between 0 and 1: ";
            double inp;
            cin >> inp;
            cout << mlp.predict({ inp }) << endl;
        }

    }
    catch (exception e)
    {
        cout << e.what();
    }
    return 0;
}
