#include <iostream>
#include "Neuron.h"

using namespace dag;

int main() {

    Node ten;
    ten.grad_ = 1;

    ten.ComputeGradients();

    std::cout << ten.children_.size() << std::endl;
    std::cout << ten.children_[0]->grad_ << std::endl;
    std::cout << ten.children_[1]->grad_ << std::endl;
    std::cout << ten.grad_ << std::endl;

    return 0;
}
