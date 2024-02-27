#include <iostream>
#include "Activations.h"
#include "Layer.h"

int main() {

    nn::Layer test(3, 3);
    Matrix matrix;
    matrix.resize(9, 3);
    matrix.setRandom();
    test.ForwardPass(matrix);
    std::cout << test.out_ << std::endl;

    return 0;
}
