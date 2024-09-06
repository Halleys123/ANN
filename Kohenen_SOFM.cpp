#include <iostream>
//#include <MLP.cpp>
#include <Neuron.cpp>
#include <random>

using namespace std;

double generate_random_double(double lower_bound, double upper_bound) {
    // Use random_device to seed the random number generator
    std::random_device rd;

    // Mersenne Twister 19937 generator seeded with random_device
    std::mt19937 gen(rd());

    // Uniform distribution to generate values in the range [lower_bound, upper_bound]
    std::uniform_real_distribution<> distr(lower_bound, upper_bound);

    // Generate and return the random double
    return distr(gen);
}

int main()
{
    try
    {
      /*  double lower = -10.0;
        double upper = 10.0;
        const int layer_count = 2;
        const int total_presentation = 5;
        
        const int total_input_nodes = 2;
        const int total_ouput_nodes = 1;


        double** inputs = new double* [total_presentation];
        for (int i = 0; i < total_presentation; i++) {
            inputs[i] = new double[total_input_nodes];
        }
        double** desired_outputs = new double* [total_presentation];
        for (int i = 0; i < total_presentation; i++) {
            desired_outputs[i] = new double[total_ouput_nodes];
        }


        int* nodes_per_layer = new int[layer_count];
        
        nodes_per_layer[0] = total_input_nodes;
        nodes_per_layer[1] = total_ouput_nodes;

        for (int i = 0; i < total_presentation; i++) {
            for (int j = 0; j < total_input_nodes; j++) {
                inputs[i][j] = -1 * generate_random_double(lower, upper);
            }
        }   
        for (int i = 0; i < total_presentation; i++) {
            for (int j = 0; j < total_ouput_nodes; j++) {
                desired_outputs[i][j] = 5;
            }
        }


        MLP<double> mlp(layer_count, nodes_per_layer);

        mlp.init_node_weights(0, 1.0, 1.0);

        mlp.input_presentations(total_presentation, inputs);
        mlp.set_output_data(total_presentation, desired_outputs);*/

        //cout << mlp.generate_output_test();
        //cout << mlp;

        Neuron<double> n(INPUT, 1);
        n.set_print_mode(NEURON_PRINT_MODES_ALL);
        Vector<double> inp(1);
        inp[0] = 1;
        Vector<double> weights(1);
        n.modify_weights(weights);
        cout << n.generate_outputs(inp);
        cout << n;
    }
    catch (exception e)
    {
        cout << e.what();
    }
    return 0;
}
