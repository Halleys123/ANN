#include <Neuron.cpp>
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

	bool initiated = true;
	//bool auto_weight_assignment = true;
	
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
		node_list[node_num_to_moidfy]->input_data(new_weights);
	}
public:
	Layer()
	{
		initiated = false;
	}
	Layer(int node_count, LAYER_TYPE layer_type, int prev_layer_node_count = 1) : current_layer_node_count(node_count), layer_type(layer_type), prev_layer_node_count(prev_layer_node_count)
	{
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
			// node_list[i]->set_print_mode(NEURON_PRINT_MODES_WEIGHTS);
			// initializing random weights for the neuron
			init_node_weights(i);
		}
	}
	void set_min_node_weight(datatype min) {
		this->min_node_weight = min;
		for (int i = 0; i < current_layer_node_count; i++) {
			init_node_weights(i);
		}
	}
	void set_max_node_weight(datatype max) {
		this->max_node_weight = max;
		for (int i = 0; i < current_layer_node_count; i++) {
			init_node_weights(i);
		}
	}
	void set_weight_bias(datatype bias) {
		this->weight_bias = bias;
		for (int i = 0; i < current_layer_node_count; i++) {
			init_node_weights(i);
		}
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