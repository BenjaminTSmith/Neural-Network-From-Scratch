#include <iostream>
#include "dval.h"
#include "matrix.h"
#include "neuron.h"

int main() {
    Neuron neuron(7);
    Matrix<Dval> input(1, 7);
    std::cout << neuron.ForwardPass(input) << std::endl;

    return 0;
}
