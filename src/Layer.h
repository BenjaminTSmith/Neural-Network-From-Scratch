#ifndef LAYER_H
#define LAYER_H

#include "Neuron.h"

class Layer {
public:
    Layer(size_t size) {
        neurons.resize(size);
    }

    std::vector<Neuron> neurons;
};


#endif // !LAYER_H
