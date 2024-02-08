#include "NN.h"

int main() {
    
    auto neural_network = new NeuralNetwork<double>(2, {3}, 1);
    neural_network->input_layer_.neurons_ = { 0.6, 0.5 };
    std::vector<double> ground_truth = { 10 };

    neural_network->ForwardPass();
    std::cout << neural_network->output_layer_.neurons_[0].out_.value_ << std::endl;
    neural_network->BackProp(ground_truth);
    std::cout << neural_network->output_layer_.neurons_[0].out_.value_ << std::endl;
    std::cout << neural_network->MSELoss(ground_truth) << std::endl;
    neural_network->ForwardProp();

    delete neural_network;

    return 0;
}
