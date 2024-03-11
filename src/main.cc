#include <iostream>
#include "Neuron.h"

using namespace nn;

int main() {

    Node one(1);
    Node ten(10);
    Node four(4);
    Node three(3);

    ten += three;
    for (auto& child : ten.children_)
        child->grad_ = 10;

    std::cout << ten.value_ << std::endl;

    std::cout << ten.children_.size() << std::endl;
    std::cout << ten.children_[0]->grad_ << std::endl;
    std::cout << ten.children_[1]->grad_ << std::endl;
    std::cout << ten.grad_ << std::endl;
    std::cout << three.grad_ << std::endl;

    return 0;
}
