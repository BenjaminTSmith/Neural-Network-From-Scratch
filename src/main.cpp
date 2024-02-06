#include "NN.h"

int main() {
    
    NeuralNetwork<double> nn(3, {1}, 2);
    // nn.printStuff();

    nn.forwardPass();
    std::cout << nn.layers[0].neurons[0].out.value << std::endl;

    return 0;
}
