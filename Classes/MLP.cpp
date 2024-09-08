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
    // pehli baar dene pe unique true hoga
    vector<int> nodes_per_layer;
    vector<Layer> layers;

    LEARNING_RULE rule = DELTA_RULE;
    MODE mode = TRAIN;

private:
    vector<double> forward_propgation(vector<double> inputs) {
        vector<double> outputs = inputs;
        for (int i = 0; i < total_layers; i++)
        {
            outputs = layers[i].compute(outputs, i == 0);
        }
        return outputs;
    }
    vector<double> forward_propgation(vector<double> inputs, vector<vector<double>>& input_to_each_layer) {
        vector<double> outputs = inputs;
        for (int i = 0; i < total_layers; i++)
        {
            outputs = layers[i].compute(outputs, i == 0);
            input_to_each_layer.push_back(outputs); // saves output from ith layer
        }
        return outputs;
    }

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
    void initialize_neurons(vector<vector<vector<double>>> weights, vector<vector<double>> bias)
    {
        for (int i = 0; i < total_layers; i++)
        {
            layers.push_back(Layer(nodes_per_layer[i], weights[i], bias[i], UNIPOLAR_SIGMOID));
        }
    }
    void train(int total_presentations, vector<vector<double>> inputs, vector<vector<double>> desired_outputs)
    {
        for (int i = 0; i < total_presentations; i++) {
            vector<double> cur_input = inputs[i];
            vector<double> cur_desired = desired_outputs[i];
            vector<vector<double>> input_to_each_layer;
            
            if (cur_input.size() != nodes_per_layer[0]) throw invalid_argument("Training cancelled due to invalid size of input vector");
            if (cur_desired.size() != nodes_per_layer[total_layers - 1]) throw invalid_argument("Training cancelled due to invalid size of desired output vector");
            forward_propgation(cur_input, input_to_each_layer);
            // backward propgation
            // correction in a node = (d - o)*f'(net_i)*X  (X is vector of input to that node this will give del W
            // for unipolar sigmoid f'(net_i) = f(net_i)(1 - f(net_i)) // actual value


        }
    }
    /*void backward_propogation(vector<double> input_to_last_l, vector<double> desired_output_from_last_layer) {
        vector<double> desired
        for (int i = total_layers - 1; i >= 0; i--) {

        }
    }*/
    vector<double> predict(vector<double> inputs)
    {
        return forward_propgation(inputs);
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