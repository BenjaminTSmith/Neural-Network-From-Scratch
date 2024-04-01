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
    std::cout << neuron.out_ << std::endl << std::endl;

    Matrix<double> mat1(3, 3);
    mat1.SetElements({ -2, 1, 4, 5, -9, 0, 2, 2, 1 });
    std::cout << mat1 << std::endl << mat1.Transpose() << std::endl;

    return 0;
}
