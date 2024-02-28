#include <iostream>
#include "Layer.h"

int main() {

    nn::Neuron test(3);
    Matrix matrix;
    matrix.resize(3, 3);
    matrix.setRandom();
    test.Activate(matrix);
    std::vector<nn::Neuron> empty_array;

    test.ComputeGradients(matrix, empty_array);

    return 0;
}
