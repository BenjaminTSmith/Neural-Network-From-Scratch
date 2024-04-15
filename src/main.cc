#include <iostream>
#include "layer.h"

int main() {
    
    Layer layer1(5, 2);
    Layer layer2(3, 5);

    Matrix<Dval> mat(2, 1);
    mat.SetElements({ 3, 7 });

    layer1.ForwardProp(mat);
    layer2.ForwardProp(layer1);
    layer2.SetGrad();
    layer2.BackProp(layer1);
    layer1.BackProp(mat);

    for (auto& neuron_ : layer2.neurons_) {
        for (auto& element_ : neuron_.weights_.elements_) {
            std::cout << element_.grad_ << std::endl;
        }
    }

    std::cout << layer1 << std::endl;
    std::cout << layer2 << std::endl;

    return 0;
}


