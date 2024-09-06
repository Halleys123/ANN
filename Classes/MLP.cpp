#include <Layer.cpp>
#include <MLP_MODE.cpp>
#include <MLP_PRINT_MODE.cpp>
#include <LEARNING_RULES_ENUM.cpp>

template <typename datatype>
class MLP {
private:
	Layer<datatype>* layers = nullptr;

	int total_layers;
	int* node_count_per_layer = nullptr;

	int total_presentations = 0;
	datatype** presentation_data = nullptr;
	datatype** desired_ouput = nullptr;

	datatype max_node_weights = 0.0;
	datatype min_node_weights = 1.0;
	datatype node_weight_tilt = 1.0;

	bool initiated = false;

	MLP_MODE mode = MLP_TRAINING;
	MLP_PRINT_MODE print_mode = MLP_FULL;
	ACTIVATION_FUNC activation_function = ACTIVATION_FUNC::U_SIGMOID;
	LEARNING_RULE learning_rule = DELTA_RULE;
private:
	bool is_initiated() {
		if (initiated) return true;
		if (total_layers < 1 || layers == nullptr) {
			return false;
		}
		if (total_presentations == 0) return false;
		for (int i = 0; i < total_layers; i++) {
			if (!(layers[i].is_initiated())) {
				initiated = false;
				return false;
			}
		}
		initiated = true;
		return true;
	}
	void init_MLP(int total_layers, int* node_count_per_layer) {
		if (total_layers < 2) throw invalid_argument("There needs to be minimum of two layers INPUT/OUTPUT");
		
		// copy value
		this->node_count_per_layer = new int[total_layers];
		for (int i = 0; i < total_layers; i++) {
			this->node_count_per_layer[i] = node_count_per_layer[i];
		}
		layers = new Layer<datatype>[total_layers];
		layers[0].init_layer(node_count_per_layer[0], INPUT, 1);
		if (total_layers > 1) {
			for (int i = 1; i < total_layers - 1; i++) {
				if (node_count_per_layer[i] < 1) throw invalid_argument("There needs to be minimum of 1 node per hidden layer");
				layers[i].init_layer(node_count_per_layer[i], HIDDEN, node_count_per_layer[i - 1]);
			}
		}

		layers[total_layers - 1].init_layer(node_count_per_layer[total_layers - 1], OUTPUT, node_count_per_layer[total_layers - 2]);
		is_initiated();
	}
public:
	MLP() {}
	MLP(int total_layers, int* node_count_per_layer) : total_layers(total_layers) {
		init_MLP(total_layers, node_count_per_layer);
	}
	MLP(int total_layers, int* node_count_per_layer, ACTIVATION_FUNC act) : total_layers(total_layers) {
		init_MLP(total_layers, node_count_per_layer);
		activation_function = act;
		for (int i = 0; i < total_layers; i++) {
			layers[i]->set_activation_function(activation_function);
		}
	}
	~MLP() {
		delete[] layers;
		delete[] node_count_per_layer;
	}
	void input_presentations(int total_presentations, datatype** presentations) {
		this->total_presentations = total_presentations;

		int input_layer_size = node_count_per_layer[0];
		presentation_data = new datatype* [total_presentations];

		for(int j = 0;j < total_presentations;j++) {
			presentation_data[j] = new datatype[input_layer_size];
			for (int i = 0; i < input_layer_size; i++) {
				presentation_data[j][i] = presentations[j][i];
			}
		}
	}
	void set_output_data(int total_presentations, datatype** presentations) {
		this->total_presentations = total_presentations;

		int input_layer_size = node_count_per_layer[0];
		desired_ouput = new datatype* [total_presentations];

		for(int j = 0;j < total_presentations;j++) {
			desired_ouput[j] = new datatype[input_layer_size];
			for (int i = 0; i < input_layer_size; i++) {
				desired_ouput[j][i] = presentations[j][i];
			}
		}
	}
	void set_print_mode(MLP_PRINT_MODE mode) {
		this->print_mode = mode;
	}
	void change_mode(MLP_MODE mode) {
		this->mode = mode;
	}
	void init_node_weights(datatype min = 0.0, datatype max = 1.0, datatype tilt = 1.0) {
		for (int i = 0; i < total_layers; i++) {
			layers[i].set_weight_params(min, max, tilt);
		}
	}
	void set_learning_rule(LEARNING_RULE rule) {
		this->learning_rule = rule;
	}
	template <typename datatype>
	friend ostream& operator<<(ostream& out, const MLP<datatype>& mlp);
};

template <typename datatype>
ostream& operator<<(ostream& out, const MLP<datatype>& mlp) {
	switch (mlp.print_mode) {
	case MLP_FULL:
		out << "MLP Information: " << endl;
		out << "Total Layers: " << mlp.total_layers << endl;
		out << "Node Count Per Layer: ";
		for (int i = 0; i < mlp.total_layers; i++) {
			out << mlp.node_count_per_layer[i] << " ";
		}
		out << endl << endl;
		for (int i = 0; i < mlp.total_layers; i++) {
			out << mlp.layers[i] << " ";
		}
		out << endl;
		break;
	case MLP_LAYERS:
		out << "Total Layers: " << mlp.total_layers << endl;
		break;
	case MLP_NODE_COUNT:
		out << "Node Count Per Layer: ";
		for (int i = 0; i < mlp.total_layers; i++) {
			out << mlp.node_count_per_layer[i] << " ";
		}
		out << endl;
		break;
	case MLP_WEIGHTS:
		out << "Weights of Layers: ";
		for (int i = 0; i < mlp.total_layers; i++) {
			out << "Layer " << i << ": ";
			out << mlp.layers[i] << " ";
		}
		out << endl;
		break;
	default:
		out << "Invalid print mode!" << endl;
		break;
	}
	return out;
}
