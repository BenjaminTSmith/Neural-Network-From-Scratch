#ifndef NEURON_H
#define NEURON_H

#include "Value.h"

using namespace DAG;

namespace nn {

class Neuron {
public:
    std::vector<Value> weights_;
    Value bias_; 

    Neuron() {}

    Neuron(size_t nins) : weights_(nins) {
        bias_.SetRandom();
        for (auto& weight_ : weights_) weight_.SetRandom();
    } 
};

}

#endif  // !NEURON_H
