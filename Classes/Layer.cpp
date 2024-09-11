#include "Layer.h"
 //def

int Layer::get_size() {
    return total_nodes;
}
vector<double> Layer::compute(vector<double> inputs, bool unique_inputs = false)
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
            outputs.push_back(neurons[i].compute({ inputs[i] }, true));
        }
    }
    return outputs;
}
vector<double> Layer::backward_propogation(vector<double> input = {}, bool output = false, vector<double> desired = {}, vector<double> error_from_next = {}, vector<double> cur_output = {}, double eta = 0.5) {
    vector<double> delta_w_prev(input.size(), 0);

    double o_i;
    double f_dash_net_i;

    for (int i = 0; i < neurons.size(); i++) {
        o_i = neurons[i].get_output();
        f_dash_net_i = (o_i * (1 - o_i));
        if (output) {
            double error = desired[i] - o_i;
            double delta = error * f_dash_net_i;
            for (int j = 0; j < neurons[i].get_weight().size(); j++) {
                // make update varibale to save update in the output neuron
                double update_weight = (eta * delta * input[j]) + neurons[i].get_weight(j);
                double update_bias = (eta * delta) + neurons[i].get_bias();

                // save delta for prev layer = delta_current_neuron * weight with previous layers some node
                delta_w_prev[j] += delta * neurons[i].get_weight(j);

                // updating current layer current neuron weights;
                neurons[i].set_weights(j, update_weight);
                neurons[i].set_bias(update_bias);
            }
        }
        else {
            double delta = error_from_next[i] * f_dash_net_i;
            for (int j = 0; j < neurons[i].get_weight().size(); j++) {
                delta_w_prev[j] += delta;
                neurons[i].set_weights(j, (eta * delta * o_i) + neurons[i].get_weight(j));
                neurons[i].set_bias(eta * delta + neurons[i].get_bias());
            }
        }
    }
    return delta_w_prev;
}