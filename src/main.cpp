#include "NN.h"

int main() {
    
    NeuralNetwork<double> nn(2, {3}, 5);
    nn.input_layer_.neurons_ = {0.6, 0.5};

    nn.ForwardPass();
    std::cout << nn.output_layer_.neurons_[0].out_.value_ << std::endl;
    nn.BackProp();
    std::cout << nn.layers_[0].neurons_[2].weights_[1].value_ << std::endl;
    nn.ForwardProp();
    std::cout << nn.layers_[0].neurons_[2].weights_[1].value_ << std::endl;

    return 0;
}
