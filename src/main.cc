#include <iostream>
#include "Activations.h"
#include "Layer.h"

int main() {

    nn::Neuron test(3);
    Matrix matrix;
    matrix.resize(3, 3);
    matrix.setRandom();
    test.Activate(matrix);
    std::vector<nn::Neuron> empty_array;

    test.ComputeWeightGradients(matrix, empty_array, 1);

    return 0;
}
