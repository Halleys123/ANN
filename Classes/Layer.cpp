#include <Neuron.cpp>
#include <ACTIVATION_FUNCTION_ENUM.cpp>
#include <LAYER_PRINT_MODE.cpp>

using namespace std;

template <typename datatype>
class Layer
{
private:
	// Used to manage how many layers there are currently.
	static int total_layers;
	// Current layers number
	int layer_number;
	// Total number of Neurons/Nodes in current node.
	int current_layer_node_count = 0;
	// As name suggests
	int prev_layer_node_count = 5;

	double min_node_weight = 0;
	double max_node_weight = 1;
	// Decides to what side is the weight vector tilted towards 
	// If < 1.0 more weights will be more towards mininmum and > 1.0 more weights will be more towards maximum
	double node_weight_tilt = 1.0; 

	Vector<datatype> outputs;

	bool initiated = true;
	//bool auto_weight_assignment = true;
	
	Vector<datatype> input_data;
	// A double pointer to hold Nodes/Neurons in current layer
	// It is 2D because Nodes are assigned on the heap
	Neuron<datatype> **node_list = nullptr;
	// Tells what type of layer it is INPUT/OUTPUT/HIDDEN.
	LAYER_TYPE layer_type = LAYER_TYPE::INPUT;
	// Debugging purposes
	LAYER_PRINT_MODE print_mode = LAYER_PRINT_MODE_ALL;

private:
	void init_node_weights(int node_num_to_moidfy) {
		Vector<datatype> new_weights(prev_layer_node_count);
		for (int i = 0; i < prev_layer_node_count; i++) {
			new_weights[i] = min_node_weight + pow(((double)rand() / RAND_MAX), node_weight_tilt) * (max_node_weight - min_node_weight);

		}
		node_list[node_num_to_moidfy]->modify_weights(new_weights);
	}
public:
	Layer()
	{
		layer_number = -1;
		initiated = false;
	}
	Layer(int node_count, LAYER_TYPE layer_type, int prev_layer_node_count = 1) : current_layer_node_count(node_count), layer_type(layer_type), prev_layer_node_count(prev_layer_node_count)
	{
		init_layer(node_count, layer_type, prev_layer_node_count, true);
	}
	void init_layer(int node_count, LAYER_TYPE layer_type, int prev_layer_node_count = 1, bool default_constructor = false) {
		if (default_constructor == false) {
			this->current_layer_node_count = node_count;
			this->layer_type = layer_type;
			this->prev_layer_node_count = prev_layer_node_count;
		}
		input_data.set_size(prev_layer_node_count);

		layer_number = Layer::total_layers;
		Layer::total_layers += 1;
		if (current_layer_node_count < 1)
		{
			initiated = false;
			throw std::invalid_argument("Nodes in a layer can't be less than 1\nYou can remove layer from Network if you want to.");
		}
		node_list = new Neuron<datatype> *[current_layer_node_count];

		for (int i = 0; i < current_layer_node_count; i++)
		{
			node_list[i] = new Neuron<datatype>(layer_type, prev_layer_node_count);
			init_node_weights(i);
		}
	}
	bool is_initiated() {
		return initiated;
	}
	void set_weight_bias(datatype bias) {
		this->weight_bias = bias;
		for (int i = 0; i < current_layer_node_count; i++) {
			init_node_weights(i);
		}
	}
	void set_weight_params(datatype min = 0.0, datatype max = 1.0, datatype tilt = 1.0) {
		if (min > max) throw invalid_argument("Minimum can't be more than maximum");
		this->max_node_weight = max;
		this->min_node_weight = min;
		this->node_weight_tilt = tilt;
		for (int i = 0; i < current_layer_node_count; i++) {
			init_node_weights(i);
		}
	}
	void set_activation_function(ACTIVATION_FUNC act, int node_num = -1) {
		if (node_num >= 0 && node_num < current_layer_node_count) {
			node_list[node_num]->set_activation_type(act);
			return;
		}
		else if (node_num == -1) {
			for (int i = 0; i < current_layer_node_count; i++) {
				this->node_list[i]->set_activation_type(act);
			}
			return;
		}
		throw invalid_argument("Invalid Node Number, No such node present in current layer");
	}
	void set_inputs(datatype* inputs) {
		if (input_data.get_size() == 0) {
			input_data.set_size(prev_layer_node_count);
		}
		for (int i = 0; i < prev_layer_node_count; i++) {
			input_data[i] = inputs[i];
		}
	}	
	void set_inputs(Vector<datatype> inputs) {
		if (input_data.get_size() == 0) {
			input_data.set_size(prev_layer_node_count);
		}
		for (int i = 0; i < prev_layer_node_count; i++) {
			input_data[i] = inputs[i];
		}
	}
	Vector<datatype> get_inputs() {
		return input_data;
	}
	Vector<datatype> get_outputs() {
		return outputs;
	}
	datatype get_net_i(int node) {
		return this->node_list[node]->get_net_i();
	}
	Vector<datatype> generate_outputs() {
		outputs.set_size(this->current_layer_node_count);
		for (int i = 0; i < this->current_layer_node_count; i++) {
			outputs[i] = node_list[i]->generate_outputs(input_data);
		}
		return outputs;
	}
	void set_print_mode(LAYER_PRINT_MODE mode) {
		this->print_mode = mode;
	}
	template <typename datatype>
	friend std::ostream &operator<<(std::ostream &out, const Layer<datatype> &layer);
};

template <typename datatype>
std::ostream& operator<<(std::ostream& out, const Layer<datatype>& layer)
{
	if(layer.print_mode == LAYER_PRINT_MODE_NONE) {
		out << "Print mode Disabled" << endl;
		return out;
	}
	if (layer.print_mode == LAYER_PRINT_MODE_NUMBER) {
		out << "Layer Number: " << layer.layer_number << endl;
	}

	if (layer.print_mode == LAYER_PRINT_MODE_WEIGHTS) {
		out << "Layer Number: " << layer.layer_number << endl;
		for (int i = 0; i < layer.current_layer_node_count; i++) {
			layer.node_list[i]->set_print_mode(NEURON_PRINT_MODES_WEIGHTS);
			out << " Weights: " << *layer.node_list[i] << endl;
		}
	}

	if (layer.print_mode == LAYER_PRINT_MODE_ALL) {
		out << "Layer Number: " << layer.layer_number << endl;
		out << "Total Nodes: " << layer.current_layer_node_count << endl;
		out << "Min Node Weight: " << layer.min_node_weight << endl;
		out << "Max Node Weight: " << layer.max_node_weight << endl;
		out << "Node Weight Tilt: " << layer.node_weight_tilt << endl;
		out << "Weights for layer range from: " << layer.min_node_weight << " - " << layer.max_node_weight << endl;
		out << "Node Weight Tilt: " << layer.node_weight_tilt << endl << endl;
		
		for (int i = 0; i < layer.current_layer_node_count; i++) {
			layer.node_list[i]->set_print_mode(NEURON_PRINT_MODES_ALL);
			out << *layer.node_list[i] << endl;
		}
	}

	if (layer.print_mode == LAYER_PRINT_MODE_TOTAL_WEIGHTS) {
		out << "Layer Number: " << layer.layer_number << endl;
		out << "Total Nodes: " << layer.current_layer_node_count << endl;
	}

	if (layer.print_mode == LAYER_PRINT_MODE_WEIGHT_RANGE) {
		out << "Layer Number: " << layer.layer_number << endl;
		out << "Weights for layer range from: " << layer.min_node_weight << " - " << layer.max_node_weight << endl;
		out << "Node Weight Tilt: " << layer.node_weight_tilt << endl;
	}

	return out;
}


template <typename T>
int Layer<T>::total_layers = 0;