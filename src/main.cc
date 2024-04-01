#include <iostream>
#include "dval.h"
#include "matrix.h"
#include "neuron.h"
#include "layer.h"

int main() {
    Neuron neuron(3);
    Matrix<Dval> input(1, 3);
    neuron.ForwardPass(input);

    std::cout << input << std::endl;
    std::cout << neuron.weights_ << std::endl << std::endl;;
    std::cout << neuron.out_ << std::endl;

    return 0;
}
