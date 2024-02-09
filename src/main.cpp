#include "NN.h"
#include <iostream>

int main() {
    
    auto neural_network = new NeuralNetwork<double>(5, {5, 5, 5, 5}, 12);
    neural_network->set_input_layer({ 0.5, 3, 10, -3, 4});
    std::vector<double> ground_truth = {
        1, -1, 0.3, -0.75, -1, 0.5, -0.3, -0.5, 0.5, 0.83, 0.0003, 0.000098
    };

    neural_network->ForwardPass();
    std::cout << "Predictions: " << std::endl;
    std::cout << neural_network->output_layer_.neurons_[0].out_.value_ << std::endl;
    std::cout << neural_network->output_layer_.neurons_[1].out_.value_ << std::endl;
    std::cout << neural_network->output_layer_.neurons_[2].out_.value_ << std::endl;
    std::cout << neural_network->output_layer_.neurons_[3].out_.value_ << std::endl;
    std::cout << neural_network->output_layer_.neurons_[4].out_.value_ << std::endl;
    std::cout << neural_network->output_layer_.neurons_[5].out_.value_ << std::endl;
    std::cout << neural_network->output_layer_.neurons_[6].out_.value_ << std::endl;
    std::cout << neural_network->output_layer_.neurons_[7].out_.value_ << std::endl;
    std::cout << neural_network->output_layer_.neurons_[8].out_.value_ << std::endl;
    std::cout << neural_network->output_layer_.neurons_[9].out_.value_ << std::endl;
    std::cout << neural_network->output_layer_.neurons_[10].out_.value_ << std::endl;
    std::cout << neural_network->output_layer_.neurons_[11].out_.value_ << std::endl;

    for (int i = 0; i < 100000; ++i) {
        neural_network->BackProp(ground_truth);
        neural_network->ForwardProp();
        neural_network->ForwardPass();
    }

    std::cout << "Final Predictions: " << std::endl;
    std::cout << neural_network->output_layer_.neurons_[0].out_.value_ << std::endl;
    std::cout << neural_network->output_layer_.neurons_[1].out_.value_ << std::endl;
    std::cout << neural_network->output_layer_.neurons_[2].out_.value_ << std::endl;
    std::cout << neural_network->output_layer_.neurons_[3].out_.value_ << std::endl;
    std::cout << neural_network->output_layer_.neurons_[4].out_.value_ << std::endl;
    std::cout << neural_network->output_layer_.neurons_[5].out_.value_ << std::endl;
    std::cout << neural_network->output_layer_.neurons_[6].out_.value_ << std::endl;
    std::cout << neural_network->output_layer_.neurons_[7].out_.value_ << std::endl;
    std::cout << neural_network->output_layer_.neurons_[8].out_.value_ << std::endl;
    std::cout << neural_network->output_layer_.neurons_[9].out_.value_ << std::endl;
    std::cout << neural_network->output_layer_.neurons_[10].out_.value_ << std::endl;
    std::cout << neural_network->output_layer_.neurons_[11].out_.value_ << std::endl;

    delete neural_network;

    return 0;
}
