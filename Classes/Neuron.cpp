#pragma once

#include <vector>
#include "print_vec.h"
#include "ACTIVATION.cpp"

using namespace std;

class Neuron
{
private:
	vector<double> weights;
	double bias = 0.0;
	double net_i = 0.0;
	double o_i = 0.0;
	ACTIVATION function = UNIPOLAR_BINARY;

public:
	Neuron() {}
	Neuron(vector<double> weights, double bias = 0.0, ACTIVATION function = UNIPOLAR_BINARY) : bias(bias), function(function)
	{
		this->weights = weights;
	}
	void set_activation(ACTIVATION func)
	{
		function = func;
	}
	void set_bias(double new_bias) {
		this->bias = new_bias;
	}
	double get_bias() {
		return bias;
	}
	void set_weights(vector<double> weights)
	{
		this->weights = weights;
	}
	void set_weights(int pos, double value)
	{
		if (pos < 0 || pos >= weights.size()) throw invalid_argument("Position is out of bounds");
		this->weights[pos] = value;
	}
	double get_output() {
		return o_i;
	}
	const vector<double> get_weight() const {
		return weights;
	}
	double get_weight(int pos) {
		if (pos < 0 || pos >= weights.size()) throw invalid_argument("Position is out of bounds");
		return weights[pos];
	}
	double get_net_i() {
		return net_i;
	}
	double compute(vector<double> inputs, bool input_layer = true)
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
	friend ostream &operator<<(ostream &os, const Neuron &n)
	{
		os << "Weights: " << n.weights << "\n";
		switch (n.function)
		{
		case UNIPOLAR_SIGMOID:
			os << "Activation Function: Unipolar Sigmoid\n";
			break;
		case BIPOLAR_SIGMOID:
			os << "Activation Function: Bipolar Sigmoid\n";
			break;
		case UNIPOLAR_BINARY:
			os << "Activation Function: Unipolar Binary\n";
			break;
		case BIPOLAR_BINARY:
			os << "Activation Function: Bipolar Binary\n";
			break;
		}
		os << "Bias: " << n.bias << "\n";
		os << "net_i: " << n.net_i << "\n";
		os << "Output: " << n.o_i << "\n";
		return os;
	}
};
