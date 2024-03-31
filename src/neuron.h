#ifndef NEURON_H
#define NEURON_H

#include <vector>

#include "dval.h"
#include "matrix.h"

struct Neuron {
    std::vector<Dval> weights_;
    Dval bias_;
    Dval out_;
    
    Neuron(int nins) : weights_(nins) {} 
    
    Dval ForwardPass(const std::vector<Dval>& inputs) {

    }

};

#endif // !NEURON_H
