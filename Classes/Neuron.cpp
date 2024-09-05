#include <Vector.cpp>
#include <string>
#include "ActivationFunctions.hpp"
#include <NEURON_PRINT_MODE_ENUM.cpp>
#include <LAYER_TYPES_ENUM.cpp>
#include "ACTIVATION_FUNCTION_ENUM.cpp"

template <typename datatype>
class Neuron
{
	static int total_nodes;
private:
	int node_number = 0;
	datatype data = NULL;
	double lambda = 0.3;
	// Saves value of output that has not been modified using activation function.
	datatype net_i = 0;
	// Saves value of output that is modified with activation function.
	datatype func_net_i = 0;

	Vector<datatype> weights;
	Vector<datatype> inputs;

	int prev_layer_size = 0; // For using in the size of weights
	bool initiated = true;

	LAYER_TYPE layer = LAYER_TYPE::INPUT;
	NEURON_PRINT_MODE print_mode = NEURON_PRINT_MODES_ALL;
	ACTIVATION_FUNC activation_type = ACTIVATION_FUNC::U_SIGMOID;

public:
	Neuron()
	{
		Neuron::total_nodes += 1;
		node_number = Neuron<datatype>::total_nodes;
		initiated = false;
	}
	Neuron(LAYER_TYPE layer, int prev_layer_size = 1) : layer(layer), prev_layer_size(prev_layer_size)
	{
		Neuron::total_nodes += 1;
		node_number = Neuron<datatype>::total_nodes;
		// Any layer even the input layer can't have size of previous layer less than 1
		// As this is logically incorrect to have size of input layer to be less than 1
		if (!initiated)
			throw std::invalid_argument("Neuron is not initialized properly.");
		if (layer == LAYER_TYPE::INPUT)
		{
			weights.set_size(1);
			weights[0] = 1;
		}
		else
		{
			if (prev_layer_size < 1)
			{
				initiated = false;
				throw std::invalid_argument("Hidden/Output layers can't be initiated without defining previous layer's size");
			}
			weights.set_size(prev_layer_size);
			for (int i = 0; i < prev_layer_size; i++)
			{
				weights[i] = (datatype)1;
			}
		}
		weights.set_print_mode(VECTOR_PRINT_MODE::VECTOR_PRINT_MODE_DATA);
	}
	void modify_weights(const Vector<datatype> &new_weights)
	{
		if (weights.get_size() == 0) {
			weights.set_size(prev_layer_size); 
		}
		weights = new_weights;
	}
	void modify_weights(datatype vect[], int size) {
		if (size != prev_layer_size) throw invalid_argument("The size of new weight vector did not match the size of current weight vector\nAborting operation.");
		if (weights.get_size() == 0) {
			weights.set_size(prev_layer_size);
		}
		for (int i = 0; i < size; i++) {
			weights[i] = vect[i];
		}
	}
	void modify_weight(datatype data, int position) {
		if (position < 0 || position >= prev_layer_size) throw out_of_range("No node at index " + to_string(position) + " is present in the last layer.");
		if (weights.get_size() == 0) {
			weights.set_size(prev_layer_size);
		}
		weights[position] = data;
	}
	datatype  generate_outputs() {
		for (int i = 0; i < prev_layer_size; i++) {
			net_i += inputs[i];
		}
		switch (activation_type) {
		case ACTIVATION_FUNC::B_SIGMOID:
			func_net_i = ActivationFunctions::b_sigmoid(net_i, lambda);
			break;
		case ACTIVATION_FUNC::U_SIGMOID:
			func_net_i = ActivationFunctions::u_sigmoid(net_i, lambda);
			break;
		case ACTIVATION_FUNC::U_BINARY:
			func_net_i = ActivationFunctions::u_binary(net_i);
			break;
		case ACTIVATION_FUNC::B_BINARY:
			func_net_i = ActivationFunctions::b_binary(net_i);
			break;
		case ACTIVATION_FUNC::SWISH:
			func_net_i = ActivationFunctions::swish(net_i);
			break;
		case ACTIVATION_FUNC::RELU:
			func_net_i = ActivationFunctions::relu(net_i);
			break;
		case ACTIVATION_FUNC::TANH:
			func_net_i = ActivationFunctions::tanh_func(net_i);
			break;
		case ACTIVATION_FUNC::LEAKY_RELU:
			func_net_i = ActivationFunctions::leaky_relu(net_i);
			break;
		case ACTIVATION_FUNC::LINEAR:
			func_net_i = ActivationFunctions::linear(net_i);
			break;
		default:
			throw std::invalid_argument("Invalid activation type");
		}
		return func_net_i;
	}
	datatype get_net_i() {
		return net_i;
	}
	datatype get_activated_net_i() {
		return func_net_i;
	}
	void input_data(Vector<datatype> input) {
		if (input.get_size() != prev_layer_size) throw invalid_argument("The size of weight vector and input vector don't match");
		if (inputs.get_size() == 0) {
			inputs.set_size(prev_layer_size);
		}
		inputs = input;
	}
	void input_data(datatype data, int from_node_num) {
		if (from_node_num < 0 || from_node_num >= prev_layer_size) throw invalid_argument("No such node is present in last layer");
		if (inputs.get_size() == 0) {
			inputs.set_size(prev_layer_size);
		}
		inputs[from_node_num] = data;
	}
	void set_print_mode(NEURON_PRINT_MODE mode)
	{
		print_mode = mode;
	}
	void set_activation_type(ACTIVATION_FUNC type) {
		activation_type = type;
	}
	template <typename datatype>
	friend std::ostream &operator<<(std::ostream &out, const Neuron<datatype> &neuron);
};

template <typename datatype>
std::ostream& operator<<(std::ostream& out, const Neuron<datatype>& neuron)
{
	if (neuron.print_mode == NEURON_PRINT_MODE::NEURON_PRINT_MODES_NONE)
	{
		out << "NEURON_PRINT_MODE set to NONE\n";
		return out;
	}
	if (neuron.print_mode == NEURON_PRINT_MODE::NEURON_PRINT_MODE_ID || neuron.print_mode == NEURON_PRINT_MODE::NEURON_PRINT_MODES_ALL)
	{
		out << "Node Number : " << neuron.node_number << endl;
	}
	if (neuron.print_mode == NEURON_PRINT_MODE::NEURON_PRINT_MODES_WEIGHTS || neuron.print_mode == NEURON_PRINT_MODE::NEURON_PRINT_MODES_ALL)
	{
		out << "Weights: " << neuron.weights;
	}

	if (neuron.print_mode == NEURON_PRINT_MODE::NEURON_PRINT_MODES_BIAS || neuron.print_mode == NEURON_PRINT_MODE::NEURON_PRINT_MODES_ALL)
	{
		out << "Bias: " << neuron.data << '\n';
	}

	if (neuron.print_mode == NEURON_PRINT_MODE::NEURON_PRINT_MODES_INPUTS || neuron.print_mode == NEURON_PRINT_MODE::NEURON_PRINT_MODES_ALL)
	{
		out << "Inputs: " << neuron.inputs;
	}

	if (neuron.print_mode == NEURON_PRINT_MODE::NEURON_PRINT_MODES_OUTPUTS || neuron.print_mode == NEURON_PRINT_MODE::NEURON_PRINT_MODES_ALL)
	{
		out << "Net Output (before activation): " << neuron.net_i << '\n';
		out << "Activated Output: " << neuron.func_net_i << '\n';
	}

	if (neuron.print_mode == NEURON_PRINT_MODE::NEURON_PRINT_MODES_ACTIVATION_FUNC || neuron.print_mode == NEURON_PRINT_MODE::NEURON_PRINT_MODES_ALL)
	{
		out << "Activation Function: ";
		switch (neuron.activation_type)
		{
		case ACTIVATION_FUNC::B_SIGMOID:
			out << "Bipolar Sigmoid\n";
			break;
		case ACTIVATION_FUNC::U_SIGMOID:
			out << "Unipolar Sigmoid\n";
			break;
		case ACTIVATION_FUNC::U_BINARY:
			out << "Unipolar Binary\n";
			break;
		case ACTIVATION_FUNC::B_BINARY:
			out << "Bipolar Binary\n";
			break;
		case ACTIVATION_FUNC::SWISH:
			out << "Swish\n";
			break;
		case ACTIVATION_FUNC::RELU:
			out << "ReLU\n";
			break;
		case ACTIVATION_FUNC::TANH:
			out << "Tanh\n";
			break;
		case ACTIVATION_FUNC::LEAKY_RELU:
			out << "Leaky ReLU\n";
			break;
		case ACTIVATION_FUNC::LINEAR:
			out << "Linear\n";
			break;
		default:
			out << "Unknown\n";
		}
	}
	out << endl;
	return out;
}

template <typename T>
int Neuron<T>::total_nodes = 0;

// Steps to use
// 1. Initiate the Neuron with a type of layer it is in ( that is INPUT, OUTPUT, HIDDEN )
// 1a. Set activation function defaults to unipolar sigmoidal
// 2. Give inputs.
// 2. Generate outputs