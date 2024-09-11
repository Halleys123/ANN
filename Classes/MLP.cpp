#pragma once

#include <vector>
#include "Layer.cpp"
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
    vector<double> output_last;

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
            input_to_each_layer.push_back(outputs);
            outputs = layers[i].compute(outputs, i == 0);
        }
        output_last = outputs;
        return outputs;
    }

public:
    MLP(int total_layers, const vector<int> &nodes_per_layer) : nodes_per_layer(nodes_per_layer), total_layers(total_layers)
    {
        if (total_layers != nodes_per_layer.size()) throw invalid_argument("Node per layer is not same as total layers please re-check them.");
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
    void train(int total_eepocs, vector<vector<double>> presentations, vector<vector<double>> desired_outputs)
    {
        double thisEpochErr = INT_MAX;

        for (int j = 0;thisEpochErr>0.01; j++) {
             thisEpochErr = 0;
            for (int i = 0; i < presentations.size(); i++) {
               vector<double> cur_input = presentations[i];
               vector<double> cur_desired = desired_outputs[i];
               vector<vector<double>> input_to_each_layer;
            
               if (cur_input.size() != nodes_per_layer[0]) throw invalid_argument("Training cancelled due to invalid size of input vector");
               if (cur_desired.size() != nodes_per_layer[total_layers - 1]) throw invalid_argument("Training cancelled due to invalid size of desired output vector");
               vector<double> err= forward_propgation(cur_input, input_to_each_layer);
               double thisPresentationErr = 0;
               for (int k = 0; k < desired_outputs[0].size(); k++) {
                   thisPresentationErr += pow(desired_outputs[i][k] - err[k], 2);
               }
               thisEpochErr += thisPresentationErr;
               backward_propogation(input_to_each_layer, cur_desired);

            }
            //cout << net_error(desired_outputs[j], total_eepocs) << endl;
            //cout << (*this).predict({0.3}) << endl;
            cout << thisEpochErr << endl;
        }
    }
    double net_error(vector<double> desired_outputs, int total_eepocs){
        double this_iteration_error = 0;
        for (int i = 0; i < desired_outputs.size(); i++) {
            this_iteration_error += pow(desired_outputs[i] - output_last[i], 2);
        }
        return this_iteration_error / total_eepocs;
    }
    void backward_propogation(vector<vector<double>> input_to_ith_layer, vector<double> desired_output_from_last_layer) {
        vector<double> change_in_weight = desired_output_from_last_layer;
        vector<double> output_from_cur = {};
    
        for (int i = total_layers - 1; i > 0; i--) {
            change_in_weight = layers[i].backward_propogation(input_to_ith_layer[i], i == total_layers - 1, desired_output_from_last_layer, change_in_weight, output_from_cur);
            output_from_cur = input_to_ith_layer[i];
            //cout << change_in_weight << endl;
        }

    }
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