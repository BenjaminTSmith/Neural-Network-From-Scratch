#include "NN.h"

int main() {
    
    NeuralNetwork<double> nn(2, {3}, 5);
    nn.inputLayer.neurons = {0.6, 0.5};

    nn.forwardPass();
    std::cout << nn.outputLayer.neurons[0].out.value << std::endl;
    nn.backProp();
    std::cout << nn.outputLayer.neurons[0].sum.grad << std::endl;
    std::cout << nn.outputLayer.neurons[0].bias.grad << std::endl;
    std::cout << nn.layers[0].neurons[0].out.grad << std::endl;
    std::cout << nn.layers[0].neurons[0].bias.grad << std::endl;
    std::cout << nn.layers[0].neurons[0].sum.grad << std::endl;
    std::cout << nn.layers[0].neurons[0].weights[0].grad << std::endl;
    std::cout << nn.layers[0].neurons[0].weights[1].grad << std::endl;

    return 0;
}
