#include <iostream>
#include "Layer.h"

int main() {

    nn::Neuron neuron(3);
    Eigen::MatrixXd input(2, 3);
    input << 5, 4, 3, 2, 1, 0;
    std::cout << input << std::endl;

    return 0;
}
