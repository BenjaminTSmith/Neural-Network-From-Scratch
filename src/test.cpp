#include "NN.h" 

int main() {

    auto nn = new NeuralNetwork<double>(800, {10}, 10);
    std::vector<double> input_layer;
    input_layer.resize(800);
    std::fill(input_layer.begin(), input_layer.end(), 200);
    nn->set_input_layer(input_layer);
    std::vector<double> ground_truth = {
        0.7, -0.3, 0.4, 0.03, 1, -0.9, -0.01, 0.1, 0.2, -0.36
    };
    nn->PrintOutput();
    for (int i = 0; i < 100; i++) {
        nn->ForwardPass();
        nn->Prop(ground_truth);
    }
    nn->PrintOutput();
    
    delete nn;

    return 0;
}
