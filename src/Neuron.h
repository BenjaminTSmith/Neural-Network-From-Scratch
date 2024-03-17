#ifndef NEURON_H
#define NEURON_H

#include <stdexcept>
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

    void ForwardPass(const std::vector<Value>& ins) {
        if (ins.size() != weights_.size()) 
            throw std::range_error("nins don't match nweights");
    }
};

}

#endif  // !NEURON_H
