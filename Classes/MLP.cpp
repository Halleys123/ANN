
#pragma once

#include "MLP.h"
// Definitions
void MLP::set_learning_rule(LEARNING_RULE rule)
{
    this->rule = rule;
}

void MLP::set_mode(MODE mode)
{
    this->mode = mode;
}

void MLP::set_learning_const(double value)
{
    this->learning_constant = value;
}
void MLP::set_error_limit(double error)
{
    this->error_limit = error;
}
void MLP::set_error_percent_print_mode(YES_NO option)
{
    this->print_error = option;
}

void MLP::initialize_neurons()
{
    vector<vector<vector<double>>> weights;
    vector<vector<double>> bias;

    for (int cur_layer_num = 0; cur_layer_num < total_layers; cur_layer_num++)
    {
        vector<vector<double>> cur_layer_weights;
        vector<double> cur_layer_bias;
        for (int cur_node_num = 0; cur_node_num < nodes_per_layer[cur_layer_num]; cur_node_num++)
        {
            double b = generateRandomNumBetween(0, 1);
            cur_layer_bias.push_back(b);
            if (cur_layer_num == 0)
            {
                cur_layer_weights.push_back({1});
            }
            else
            {
                vector<double> cur_node_weights;
                for (int prev_layer_nodes = 0; prev_layer_nodes < nodes_per_layer[cur_layer_num - 1]; prev_layer_nodes++)
                {
                    double w = generateRandomNumBetween(-1, 1);
                    cur_node_weights.push_back(w);
                }
                cur_layer_weights.push_back(cur_node_weights);
            }
        }
        weights.push_back(cur_layer_weights);
        bias.push_back(cur_layer_bias);
    }
    this->weight = weights;
    this->bias = bias;

    for (int i = 0; i < total_layers; i++)
    {
        layers.push_back(Layer(nodes_per_layer[i], weights[i], bias[i], UNIPOLAR_SIGMOID));
    }
}

void MLP::train(int total_eepocs, vector<vector<double>> presentations, vector<vector<double>> desired_outputs)
{
    double thisEpochErr = INT_MAX;

    for (int j = 0; thisEpochErr > error_limit; j++)
    {
        if (print_error == YES)
            thisEpochErr = 0;
        for (int i = 0; i < presentations.size(); i++)
        {

            vector<double> cur_input = presentations[i];
            vector<double> cur_desired = desired_outputs[i];
            vector<vector<double>> input_to_each_layer;

            if (cur_input.size() != nodes_per_layer[0])
                throw invalid_argument("Training cancelled due to invalid size of input vector");
            if (cur_desired.size() != nodes_per_layer[total_layers - 1])
                throw invalid_argument("Training cancelled due to invalid size of desired output vector");

            if (print_error == YES)
            {
                vector<double> err = forward_propgation(cur_input, input_to_each_layer);
                double thisPresentationErr = 0;
                for (int k = 0; k < desired_outputs[0].size(); k++)
                {
                    thisPresentationErr += pow(desired_outputs[i][k] - err[k], 2);
                }
                thisEpochErr += thisPresentationErr;
            }
            else
            {
                forward_propgation(cur_input, input_to_each_layer);
            }
            backward_propogation(input_to_each_layer, cur_desired);
        }
        if (print_error == YES)
            cout << thisEpochErr << endl;
    }
}
double MLP::net_error(vector<double> desired_outputs, int total_eepocs)
{
    double this_iteration_error = 0;
    for (int i = 0; i < desired_outputs.size(); i++)
    {
        this_iteration_error += pow(desired_outputs[i] - output_last[i], 2);
    }
    return this_iteration_error / total_eepocs;
}

void MLP::backward_propogation(vector<vector<double>> input_to_ith_layer, vector<double> desired_output_from_last_layer)
{
    vector<double> change_in_weight = desired_output_from_last_layer;
    vector<double> output_from_cur = {};

    for (int i = total_layers - 1; i > 0; i--)
    {
        change_in_weight = layers[i].backward_propogation(input_to_ith_layer[i], i == total_layers - 1, desired_output_from_last_layer, change_in_weight, output_from_cur, learning_constant);
        output_from_cur = input_to_ith_layer[i];
    }
}
vector<double> MLP::predict(vector<double> inputs)
{
    return forward_propgation(inputs);
};
