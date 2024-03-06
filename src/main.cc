#include <iostream>
#include "Node.h"

int main() {

    nn::Node test1(3);
    nn::Node test2(4);
    test2 += test1;
    for (const auto& child : test2.children_) {
        std::cout << *child << std::endl;
    }
    std::cout << "test2: " << test2 << std::endl;


    return 0;
}
