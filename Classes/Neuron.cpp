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
	void set_weights(vector<double> weights)
	{
		this->weights = weights;
	}
	double get_weight(int pos) {
		return weights[pos];
	}
	double compute(vector<double> inputs)
	{
		if (inputs.size() != weights.size()) {
			throw invalid_argument("Input size does not match weight size");
		}

		net_i = 0.0;
		for (int i = 0; i < inputs.size(); i++)
		{
			net_i += inputs[i] * weights[i];
		}
		net_i += bias;
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
		os << "Net Input: " << n.net_i << "\n";
		os << "Output: " << n.o_i << "\n";
		return os;
	}
};
