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
	void set_activation(ACTIVATION func);
	void set_bias(double new_bias);
	void set_weights(vector<double> weights);
	void set_weights(int pos, double value);

	const vector<double> get_weight() const;
	double get_weight(int pos);
	double get_output();
	double get_net_i();
	double get_bias();

	double compute(vector<double> inputs, bool input_layer = true);

	friend ostream& operator<<(ostream& os, const Neuron& n)
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
