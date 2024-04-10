#include "dval.h"
#include "neuron.h"
#include "matrix.h"
#include <iostream>

int main() {
    Neuron neuron(3);
    Matrix<Dval> mat(3, 2);
    mat.SetElements({2, -1,  0, 6,  1, 0});
    neuron.ForwardProp(mat);
    neuron.ZeroGrad();
    neuron.out_[0].grad_ = 1;
    neuron.out_[1].grad_ = 1;
    neuron.BackProp(mat);

    std::cout << mat << std::endl;
    for (auto& element_ : neuron.weights_.elements_)
        std::cout << element_.grad_ << std::endl;

    return 0;
}
