#include "NN.h"

int main() {
    
    NeuralNetwork<double> nn(2, {1}, 1);
    nn.inputLayer.neurons = {-0.05, 0.6};

    nn.forwardPass();
    std::cout << nn.layers[0].neurons[0].weights.size() << std::endl;
    std::cout << nn.outputLayer.neurons[0].out.value << std::endl;

    return 0;
}
