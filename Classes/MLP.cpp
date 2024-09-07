#pragma once

#include <vector>
#include <Layer.cpp>
#include "LEARNING_RULE.cpp"
#include "MODE.cpp"
#include "print_vec.h"

using namespace std;

class MLP
{
private:
    int total_layers = 0;
    int total_presentations = 0;
    // pehli baar dene pe unique true hoga
    vector<vector<double>> inputs;
    vector<vector<double>> desired_outputs;
    vector<vector<vector<double>>> weights;
    vector<vector<double>> bias;

    vector<int> nodes_per_layer;
    vector<Layer> layers;

    LEARNING_RULE rule = DELTA_RULE;
    MODE mode = TRAIN;

public:
    MLP(int total_layers, const vector<int> &nodes_per_layer) : nodes_per_layer(nodes_per_layer), total_layers(total_layers)
    {
    }
    void set_learning_rule(LEARNING_RULE rule)
    {
        this->rule = rule;
    }
    void set_mode(MODE mode)
    {
        this->mode = mode;
    }
    void set_presentations(int total_presentations, vector<vector<double>> inputs, vector<vector<double>> desired_outputs)
    {
        this->total_presentations = total_presentations;
        this->inputs = inputs;
        this->desired_outputs = desired_outputs;
    }
    void initialize_neurons(vector<vector<vector<double>>> weights, vector<vector<double>> bias)
    {
        for (int i = 0; i < total_layers; i++)
        {
            layers.push_back(Layer(nodes_per_layer[i], weights[i], bias[i], UNIPOLAR_SIGMOID));
        }
    }
    vector<double> predict(vector<double> inputs)
    {
        vector<double> outputs = inputs;
        for (int i = 0; i < total_layers; i++)
        {
            outputs = layers[i].compute(outputs, i == 0);
        }
        return outputs;
    }
    friend ostream &operator<<(ostream &os, const MLP &mlp)
    {
        for (int i = 0; i < mlp.total_layers; i++)
        {
            os << mlp.layers[i] << endl;
        }
        return os;
    }
};