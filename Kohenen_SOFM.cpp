#include <iostream>
#include <MLP.cpp>
#include "print_vec.h"

using namespace std;

int main()
{
    try
    {
        int total_presentations = 1;
        // same as desired
        vector<vector<double>> inputs = {
            {0.2}, {0.25}, {0.4} };
        // outer most layer represent presentaion number
        // in that is an array showing ouputs from various nodes
        vector<vector<double>> desired = {
            {0.4}, {0.5} };
        // weights inner tells number of neurons in last layer or weights connected to prev layers each neuron from
        // given neuron
        // layer after that tells about layer number
        vector<vector<vector<double>>> weights = {
            {{1}, {1}, {1}},
            {{0.2, 0.3, 0.3}, {0.2, 0.3, 0.3}, {0.2, 0.3, 0.3}, {0.2, 0.3, 0.3}},   
            {{0.1, 0.3, 0.2, 0.5},{0.1, 0.3, 0.2, 0.5},{0.1, 0.3, 0.2, 0.5},{0.1, 0.3, 0.2, 0.5}},
            {{ 0.1, 0.3, 0.2, 0.5 }, {0.1, 0.3, 0.2, 0.5}, {0.1, 0.3, 0.2, 0.5}, {0.1, 0.3, 0.2, 0.5}, {0.1, 0.3, 0.2, 0.5}, {0.1, 0.3, 0.2, 0.5}, {0.1, 0.3, 0.2, 0.5}, {0.1, 0.3, 0.2, 0.5}, {0.1, 0.3, 0.2, 0.5}, {0.1, 0.3, 0.2, 0.5} }
        };
        // bias outer layers defines a layer inner layer tells bias of each node
        vector<vector<double>> bias = { {0.0, 0.2, 0.3}, {0.6, 0.2, 0.2, 0.3}, {0.1, 0.7, 0.5, 0.6} , {0.7, 0.6,0.7, 0.6,0.7, 0.6,0.7, 0.6,0.7, 0.6} };

        vector<int> nodes_per_layer = {3, 4, 4, 10};
        MLP mlp(4, nodes_per_layer);
        mlp.train(total_presentations, inputs, desired);
        mlp.initialize_neurons(weights, bias);

        vector<double> input_test = {0, 0, 0};
        cout << mlp;
        cout << mlp.predict(input_test);
    }
    catch (exception e)
    {
        cout << e.what();
    }
    return 0;
}
