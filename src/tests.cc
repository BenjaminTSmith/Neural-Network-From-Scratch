#include <iostream>
#include "Layer.h"

// simple backprop of one input
void NeuronTest1() {
    Matrix matrix(1, 3);
    matrix.setRandom();
    std::cout << "Input values: " << matrix << std::endl << std::endl;
    nn::Neuron test(3);
    test.Activate(matrix);
    test.activated_out_ = Matrix::Ones(test.activated_out_.rows(), test.activated_out_.cols());
    std::cout << test.out_ << std::endl << std::endl;
    std::vector<nn::Neuron> empty_vec;
    std::cout << test.weights_ << std::endl << std::endl;
    test.ComputeGradients(matrix, empty_vec);
    std::cout << test.weights_ << std::endl << std::endl;
}

void LayerTest1() {
    nn::Layer layer(3, 2);
    Matrix input(1, 2);
}

int main() {

    NeuronTest1();

    return 0;
}
