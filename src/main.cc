#include <iostream>
#include "Node.h"

int main() {

    nn::Node test1(3);
    nn::Node test2(4);
    test2 += test1;
    test2.ReLU();
    for (const auto& child : test2.children_) {
        for (const auto& child : child->children_) {
            std::cout << *child << std::endl;
        }
        std::cout << *child << std::endl;
    }
    std::cout << "test2: " << test2 << std::endl;

    test2.grad_ = 1;
    test2.ComputeGradients();
    std::cout << test2.children_[0]->grad_ << std::endl;


    return 0;
}
