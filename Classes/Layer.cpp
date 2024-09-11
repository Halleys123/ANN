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
        vector<double> outputs;
        for (int i = 0; i < total_nodes; i++)
        {
            if (!unique_inputs)
            {
                    outputs.push_back(neurons[i].compute(inputs, false));
            }
            else
            {
                    outputs.push_back(neurons[i].compute({inputs[i]}, true));
            }
            //cout << neurons[i] << endl << endl; 
        }
        return outputs;
    }
    vector<double> backward_propogation(vector<double> input = {}, bool output = false, vector<double> desired = {}, vector<double> error_from_next = {}, vector<double> cur_output = {}) {
        vector<double> delta_w_prev(input.size(), 0);

        double eta = 0.5;
        double o_i;
        double f_dash_net_i;

        for (int i = 0; i < neurons.size();i++) {
            o_i = neurons[i].get_output();
            f_dash_net_i = (o_i * (1 - o_i));
            if (output) {
                double error = desired[i] - o_i;
                double delta = error * f_dash_net_i;
                //cout << "Output: " << o_i << endl;
                /*cout << "Layer " << i + 1 << endl;
                cout << "Delta: " << delta_w_prev << endl;*/
                for (int j = 0; j < neurons[i].get_weight().size(); j++) {
                    // make update varibale to save update in the output neuron
                    double update_weight = (eta * delta * input[j]) + neurons[i].get_weight(j);
                    double update_bias = (eta * delta) + neurons[i].get_bias();
                    
                    // save delta for prev layer = delta_current_neuron * weight with previous layers some node
                    delta_w_prev[j] += delta * neurons[i].get_weight(j);
                    //cout << "Delta update " << j + 1 << " " << delta_w_prev << endl;
                    // updating current layer current neuron weights;
                    neurons[i].set_weights(j, update_weight);
                    neurons[i].set_bias(update_bias);
                }
            }
            else {
                //cout << endl;
                double delta = error_from_next[i] * f_dash_net_i;
               /* cout << "Neuron " << i + 1 << endl << "Error: " << delta << endl;
                cout << "Output: " << o_i << endl;
                cout << "Error from next" << error_from_next[i] << endl;
                cout << "F dash net i: " << f_dash_net_i << endl;*/
                for (int j = 0; j < neurons[i].get_weight().size(); j++) {
                    delta_w_prev[j] += delta;
                    neurons[i].set_weights(j, (eta * delta * o_i) + neurons[i].get_weight(j));
                    neurons[i].set_bias(eta * delta + neurons[i].get_bias());
                }
            }
            //cout << endl;
        }
        return delta_w_prev;
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