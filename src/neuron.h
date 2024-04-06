#ifndef NEURON_H
#define NEURON_H

#include "dval.h"
#include "matrix.h"

struct Neuron {
    Matrix<Dval> weights_;
    Dval bias_;
    Matrix<Dval> out_;
    
    Neuron(int nins) 
        : weights_(1, nins),
          bias_(0) {} 

    Neuron() {}

    Matrix<Dval> ForwardProp(const Matrix<Dval>& inputs) {
        // weights_ is a row vector, inputs need to be columns
        out_ = weights_ * inputs + bias_;
        return out_;
    }

    void BackProp(const Matrix<Dval>& inputs) {} 
        
};

#endif // !NEURON_H
