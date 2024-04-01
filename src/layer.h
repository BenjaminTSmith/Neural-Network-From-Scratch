#ifndef LAYER_H
#define LAYER_H 

#include "neuron.h"

class Layer {
    std::vector<Neuron> neurons_;

    Layer(int nneurons, int nins) : neurons_(nneurons) {
        for (auto& neuron_ : neurons_) neuron_ = Neuron(nins);
    }
};

#endif // !LAYER_H
