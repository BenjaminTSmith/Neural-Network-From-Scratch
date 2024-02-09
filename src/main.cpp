#include "NN.h"

int main() {
    
    auto neural_network = new NeuralNetwork<double>(5, {3}, 5);
    neural_network->input_layer_.neurons_ = { 0.5, 3, 10, -3, 4 };
    std::vector<double> ground_truth = {1, -1, 0.3, -0.75, -1};

    neural_network->ForwardPass();
    std::cout << neural_network->output_layer_.neurons_[0].out_.value_ << std::endl;
    std::cout << neural_network->output_layer_.neurons_[1].out_.value_ << std::endl;
    std::cout << neural_network->output_layer_.neurons_[2].out_.value_ << std::endl;
    std::cout << neural_network->output_layer_.neurons_[3].out_.value_ << std::endl;
    std::cout << neural_network->output_layer_.neurons_[4].out_.value_ << std::endl;

    for (int i = 0; i < 1000; ++i) {
        neural_network->BackProp(ground_truth);
        neural_network->ForwardProp();
        neural_network->ForwardPass();
    }

    std::cout << neural_network->output_layer_.neurons_[0].out_.value_ << std::endl;
    std::cout << neural_network->output_layer_.neurons_[1].out_.value_ << std::endl;
    std::cout << neural_network->output_layer_.neurons_[2].out_.value_ << std::endl;
    std::cout << neural_network->output_layer_.neurons_[3].out_.value_ << std::endl;
    std::cout << neural_network->output_layer_.neurons_[4].out_.value_ << std::endl;

    delete neural_network;

    return 0;
}
