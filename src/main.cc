#include <iostream>
#include "Neuron.h"

using namespace nn;

int main() {

    Neuron test(3);

    std::vector<Node> inputs = {7, 2, -3};

    test.ForwardPass(inputs);
    test.out_.grad_ = 1;
    test.BackProp();
    for (auto& child : test.out_.children_)
        std::cout << child->grad_ << std::endl;
    std::cout << test.bias_.grad_ << std::endl;

    return 0;
}
