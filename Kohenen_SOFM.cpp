#include <iostream>
#include <Layer.cpp>

using namespace std;

// Make a timelapse programming Kohenen self organizing feature map in cpp

int main()
{
   /* Layer<double> layer(10, LAYER_TYPE::HIDDEN, 7);
    layer.set_max_node_weight(10);
    Layer<double> laye1(5, LAYER_TYPE::HIDDEN, 10);
    cout << layer;
    cout << laye1;*/
    try {
        /*Neuron<double> neuron(LAYER_TYPE::HIDDEN, 2);
        neuron.set_activation_type(ACTIVATION_FUNC::SWISH);*/
        //neuron.set_print_mode(/*NEURON_PRINT_MODE_ID*/);
        /*Vector<double> inputs(2);
        inputs[0] = 5;
        inputs[1] = 5;
        neuron.modify_weight(0.3, 0);
        neuron.modify_weight(0.6, 1);

        neuron.input_data(inputs);
        neuron.generate_outputs();
        cout << neuron;
        cout << neuron.get_net_i() << endl;
        cout << neuron.get_activated_net_i();*/

        Layer<double> layer(5, LAYER_TYPE::HIDDEN, 5);
        layer.set_max_node_weight(5.0);
        cout << layer;
    }
    catch (exception e) {


        cout << e.what();
    }
    return 0;
}
