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
    vector<double> backward_propogation() {
        // this will return desired output for previous layer

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