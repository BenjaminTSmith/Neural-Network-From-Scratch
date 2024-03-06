#ifndef LAYER_H
#define LAYER_H

#include "Neuron.h"

namespace nn {

class Layer {
public:
    std::vector<Neuron> neurons_;
    
    Layer(int size, int prev_layer_size) : neurons_(size) {
        for (auto& neuron : neurons_) neuron = Neuron(prev_layer_size);
    }

    void ForwardPass(Layer& prev_layer) {

    }

    void BackProp() {
    }

};

}

#endif // !LAYER_H
