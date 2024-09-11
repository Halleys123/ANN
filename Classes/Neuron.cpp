#include "Neuron.h"
// Def
void Neuron::set_activation(ACTIVATION func)
{
	function = func;
}
void Neuron::set_bias(double new_bias) {
	this->bias = new_bias;
}
double Neuron::get_bias() {
	return bias;
}
void Neuron::set_weights(vector<double> weights)
{
	this->weights = weights;
}
void Neuron::set_weights(int pos, double value)
{
	if (pos < 0 || pos >= weights.size()) throw invalid_argument("Position is out of bounds");
	this->weights[pos] = value;
}
double Neuron::get_output() {
	return o_i;
}
const vector<double> Neuron::get_weight() const {
	return weights;
}
double Neuron::get_weight(int pos) {
	if (pos < 0 || pos >= weights.size()) throw invalid_argument("Position is out of bounds");
	return weights[pos];
}
double Neuron::get_net_i() {
	return net_i;
}
double Neuron::compute(vector<double> inputs, bool input_layer)
{
	if (inputs.size() != weights.size()) {
		throw invalid_argument("Input size does not match weight size");
	}
	net_i = 0.0;
	for (int i = 0; i < inputs.size(); i++)
	{
		net_i += inputs[i] * weights[i];
		//cout << net_i << " + " << inputs[i] << " * " << weights[i] << " = " << net_i << endl;
	}
	net_i += bias;
	if (input_layer) {
		o_i = net_i;
		return o_i;
	}
	switch (function)
	{
	case UNIPOLAR_SIGMOID:
		o_i = 1 / (1 + exp(-net_i));
		break;
	case BIPOLAR_SIGMOID:
		o_i = (2 / (1 + exp(-net_i))) - 1;
		break;
	case UNIPOLAR_BINARY:
		o_i = net_i > 0 ? 1 : 0;
		break;
	case BIPOLAR_BINARY:
		o_i = net_i > 0 ? 1 : -1;
		//case ADAM:
			//o_i = 
		break;
	}
	return o_i;
}