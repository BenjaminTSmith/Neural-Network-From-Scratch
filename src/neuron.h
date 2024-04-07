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
        out_ = out_.Max(0).Transpose();
        return out_;
    }

    void BackProp(const Matrix<Dval>& inputs) {
        // bias_.grad_ = out_.grad_;
        for (size_t i = 0; i < weights_.elements_.size(); i++) {
            weights_[0].grad_ += inputs.RowwiseAverage()[0].value_;
        }
    } 

    void ZeroGrad() {
        for (auto& element_ : out_.elements_) element_.grad_ = 0;
        bias_.grad_ = 0;
        for (auto& element_ : weights_.elements_) element_.grad_ = 0;
    }
        
};

#endif // !NEURON_H
