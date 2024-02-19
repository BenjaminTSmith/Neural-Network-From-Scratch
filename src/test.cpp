#include "NN.h" 

int main() {

    auto nn = new NeuralNetwork<double>(10, {10, 10}, 10);
    std::vector<double> input_layer;
    input_layer.resize(10);
    std::fill(input_layer.begin(), input_layer.end(), 3);
    nn->set_input_layer(input_layer);
    std::vector<double> ground_truth = {
        0, 1, 0, 0, 0, 0, 0, 0, 0
    };
    nn->PrintOutput();
    for (int i = 0; i < 10000; i++) {
        nn->ForwardPass();
        nn->Prop(ground_truth);
        // nn->PrintOutput();
    }
    nn->PrintOutput();
    
    delete nn;

    return 0;
}
