#include "NN.h" 

int main() {

    auto nn = new NeuralNetwork<double>(1000, {10, 10}, 10);
    std::vector<double> input_layer;
    input_layer.resize(1000);
    std::fill(input_layer.begin(), input_layer.end(), 0);
    nn->set_input_layer(input_layer);
    std::vector<double> ground_truth = {
        0.7, 0.3, 0.4, 0.03, 1, -0.8, -0.2, 0.5, -3, 2
    };
    nn->PrintOutput();
    for (int i = 0; i < 1000; i++) {
        nn->ForwardPass();
        nn->Prop(ground_truth);
        // nn->PrintOutput();
    }
    nn->SoftMax();
    nn->PrintOutput();
    
    delete nn;

    return 0;
}
