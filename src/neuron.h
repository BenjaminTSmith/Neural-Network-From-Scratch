#ifndef NEURON_H
#define NEURON_H

#include "dval.h"
#include "matrix.h"

struct Neuron {
    Matrix<Dval> weights_;
    Dval bias_;
    Matrix<Dval> out_;
    double learning_rate_ = 0.01;

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
        bias_.grad_ = out_.ColwiseAverage()[0].grad_;
        for (size_t i = 0; i < inputs.row_count_; i++) {
            for (size_t j = 0; j < inputs.col_count_; j++) {
                weights_[i].grad_ += out_[i].grad_ 
                    * inputs[i * inputs.row_count_ + j].grad_;
            }
        }

        bias_.value_ -= bias_.grad_ * learning_rate_ ;
        for (auto& element_ : weights_.elements_) {
            element_.value_ -= element_.grad_ * learning_rate_
                * (1 / static_cast<double>(inputs.col_count_));
        }
    }

    void ZeroGrad() {
        for (auto& element_ : out_.elements_) element_.grad_ = 0;
        bias_.grad_ = 0;
        for (auto& element_ : weights_.elements_) element_.grad_ = 0;
    }

};

#endif // !NEURON_H
