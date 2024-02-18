#include "NN.h" 

void SoftMax(std::vector<double>& inputs) {
    double max_input = *std::max_element(inputs.begin(), inputs.end());
    
    // Calculate the softmax values by subtracting the max value
    double summation = 0.0;
    for (auto& input : inputs) {
        input = std::exp(input - max_input); // Subtract max_input
        summation += input;
    }

    // Normalize the softmax values
    for (auto& input: inputs) {
        input /= summation;
    }
}

int main() {

    /*auto nn = new NeuralNetwork<double>(1000, {10, 10}, 11);
    std::vector<double> input_layer;
    input_layer.resize(1000);
    std::fill(input_layer.begin(), input_layer.end(), 0);
    nn->set_input_layer(input_layer);
    std::vector<double> ground_truth = {
        0.7, 0.3, 0.4, 0.03, 1, -0.8, -0.2, 10000, -30, 21.90784, 1
    };
    nn->PrintOutput();
    for (int i = 0; i < 1000; i++) {
        nn->ForwardPass();
        nn->Prop(ground_truth);
        // nn->PrintOutput();
    }
    nn->PrintOutput();
    
    delete nn;*/

    std::vector<double> test_layer = {0, 100000000, 2, 3, 4, 5};

    for (const auto& neuron: test_layer) { std::cout << neuron << std::endl; }
    SoftMax(test_layer);
    std::cout << std::endl;
    for (const auto& neuron: test_layer) { std::cout << neuron << std::endl; }

    return 0;
}
