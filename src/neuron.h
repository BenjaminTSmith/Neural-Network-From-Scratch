#ifndef NEURON_H
#define NEURON_H

#include "dval.h"
#include "matrix.h"

struct Neuron {
    Matrix<Dval> weights_;
    Dval bias_;
    Matrix<Dval> out_;
    
    Neuron(int nins) 
        : weights_(nins, 1),
          bias_(0) {} 

    Neuron() {}

    Matrix<Dval> ForwardPass(const Matrix<Dval>& inputs) {
        out_ = inputs * weights_ + bias_;
        return out_;
    }

    void BackProp(const Matrix<Dval>& inputs) {
    } 

};

#endif // !NEURON_H
