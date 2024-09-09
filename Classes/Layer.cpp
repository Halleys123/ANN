#pragma once

#include <vector>
#include <string>
#include <Neuron.cpp>
#include "print_vec.h"

class Layer
{
private:
    vector<Neuron> neurons;
    int total_nodes = 0;

public:
    Layer() {}
    Layer(int total_nodes, vector<vector<double>> weights, vector<double> bias, ACTIVATION function)
    {
        this->total_nodes = total_nodes;
        for (int i = 0; i < total_nodes; i++)
        {
            neurons.push_back(Neuron(weights[i], bias[i], function));
        }
    }
    int get_size() {
        return total_nodes;
    }
    vector<double> compute(vector<double> inputs, bool unique_inputs = false)
    {
        // isko nahi likhenge then the code breaks don't move it. !!!!!!!important - What happens - if wrong sized input is given then program breaks;
        if (unique_inputs && inputs.size() != total_nodes) throw invalid_argument("Input Vector size should be " + to_string(total_nodes) + " but is " + to_string(inputs.size()));
        if (!unique_inputs)
        {
            vector<double> outputs;
            for (int i = 0; i < total_nodes; i++)
            {
                outputs.push_back(neurons[i].compute(inputs));
            }
            return outputs;
        }
        else
        {
            vector<double> outputs;
            for (int i = 0; i < total_nodes; i++)
            {
                outputs.push_back(neurons[i].compute({inputs[i]}));
            }
            return outputs;
        }
    }
    vector<double> backward_propogation(vector<double> input, vector<double> desired_output, bool is_output_layer = false, Layer* next_layer = nullptr, vector<double> errors_from_next_layer) {
        vector<double> errors;
        double eta = 0.2;
        for (int i = 0; i < neurons.size();i++) {
            if (is_output_layer) {
                double actual_output = neurons[i].get_output();
                double error = desired_output[i] - actual_output;
                double correction = error * (actual_output * (1 - actual_output));
                vector<double> updated_weights;
                for (int j = 0; j < input.size(); j++) {
                    updated_weights.push_back(input[i] * correction * 0.2);
                }
                neurons[i].set_weights(updated_weights);
                errors.push_back(correction); // delta learning rule
            }
            else {
                double actual_output = neurons[i].get_output();
                double f_dash_net_i = (actual_output * (1 - actual_output));
                double del = 0.0;
                int next_layer_size = next_layer->get_size();
                for (int j = 0; j < next_layer_size; j++) {
                    del += next_layer->get_weight_from_node(j, i) * errors_from_next_layer[j];
                }
                vector<double> updated_weights;
                for (int j = 0; j < input.size(); j++) {
                    updated_weights.push_back(input[i] * f_dash_net_i * del * 0.2);
                }
                neurons[i].set_weights(updated_weights);
                errors.push_back(del);
            }
        }

        return errors;
    }
    double get_weight_from_node(int i, int pos) {
        return neurons[i].get_weight(pos);
    }
    friend ostream &operator<<(ostream &os, const Layer &l)
    {
        for (int i = 0; i < l.total_nodes; i++)
        {
            os << l.neurons[i] << endl;
        }
        return os;
    }
};