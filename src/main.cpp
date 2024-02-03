#include "Neuron.h"
#include <vector>
#include <iostream>

int main() {
    
    std::vector<Node> inputs = {{2}, {0}};
    std::vector<Node> weights = {{-3}, {1}};
    Neuron test(2);
    test.bias = 6.8814782;
    test.weights = weights;

    test.forwardPass(inputs);
    std::cout << test.out.value << std::endl;

    test.out.grad = 1;
    test.backProp(inputs);
    std::cout << inputs[0].grad << ' ' << inputs[1].grad << std::endl;
    std::cout << test.weights[0].grad << ' ' << test.weights[1].grad << std::endl;

    return 0;
}
