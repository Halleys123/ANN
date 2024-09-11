#pragma once

#include "data_generator.h"
#include "print_vec.h"
#include "Layer.h"

#include <vector>
#include <random>

#include "LEARNING_RULE.cpp"
#include "YES_NO_ENUM.cpp"
#include "MODE.cpp"

using namespace std;

class MLP
{
private:
    int total_layers = 0;
    double learning_constant = 0.5;
    double error_limit = 0.01;
    // pehli baar dene pe unique true hoga
    vector<int> nodes_per_layer;
    vector<Layer> layers;
    vector<double> output_last;
    vector<vector<vector<double>>> weight;
    vector<vector<double>> bias;

    LEARNING_RULE rule = DELTA_RULE;
    YES_NO print_error = NO;
    MODE mode = TRAIN;

private:
    vector<double> forward_propgation(vector<double> inputs)
    {
        vector<double> outputs = inputs;
        for (int i = 0; i < total_layers; i++)
        {
            outputs = layers[i].compute(outputs, i == 0);
        }
        return outputs;
    }
    vector<double> forward_propgation(vector<double> inputs, vector<vector<double>> &input_to_each_layer)
    {
        vector<double> outputs = inputs;
        for (int i = 0; i < total_layers; i++)
        {
            input_to_each_layer.push_back(outputs);
            outputs = layers[i].compute(outputs, i == 0);
        }
        output_last = outputs;
        return outputs;
    }

public:
    MLP(int total_layers, const vector<int> &nodes_per_layer) : nodes_per_layer(nodes_per_layer), total_layers(total_layers)
    {
        if (total_layers != nodes_per_layer.size())
            throw invalid_argument("Node per layer is not same as total layers please re-check them.");
        initialize_neurons();
    }

    void set_mode(MODE mode);
    void set_learning_rule(LEARNING_RULE rule);
    void set_learning_const(double value);
    void set_error_limit(double error);
    void set_error_percent_print_mode(YES_NO option);

    void initialize_neurons();
    void train(int total_eepocs, vector<vector<double>> presentations, vector<vector<double>> desired_outputs);
    void backward_propogation(vector<vector<double>> input_to_ith_layer, vector<double> desired_output_from_last_layer);
    double net_error(vector<double> desired_outputs, int total_eepocs);
    vector<double> predict(vector<double> inputs);

    friend ostream &operator<<(ostream &os, const MLP &mlp)
    {
        for (int i = 0; i < mlp.total_layers; i++)
        {
            os << mlp.layers[i] << endl;
        }
        return os;
    }
};