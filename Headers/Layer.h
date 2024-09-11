#pragma once

#include "print_vec.h"
#include "Neuron.h"

#include <vector>
#include <string>

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
    int get_size();

    vector<double> compute(vector<double> inputs, bool unique_inputs = false);
    vector<double> backward_propogation(vector<double> input = {}, bool output = false, vector<double> desired = {}, vector<double> error_from_next = {}, vector<double> cur_output = {}, double eta = 0.5);

    friend ostream &operator<<(ostream &os, const Layer &l)
    {
        for (int i = 0; i < l.total_nodes; i++)
        {
            os << l.neurons[i] << endl;
        }
        return os;
    }
};
