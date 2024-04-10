#ifndef NEURON_H
#define NEURON_H

#include "dval.h"
#include "matrix.h"

struct Neuron {
    Matrix<Dval> weights_;
    Dval bias_;
    Matrix<Dval> out_;
    double learning_rate_ = 1;

    Neuron(int nins) : weights_(1, nins), bias_(0) {}

    Neuron() {}

    Matrix<Dval> ForwardProp(const Matrix<Dval>& inputs) {
        // weights_ is a row vector, input needs to be a column vector or
        // a matrix with columns as inputs
        out_ = weights_ * inputs + bias_;
        out_ = out_.Max(0).Transpose();
        return out_;
    }

    void BackProp(const Matrix<Dval>& inputs) {
        for (size_t i = 0; i < out_.size(); i++) {
            bias_.grad_ += out_[i].grad_;
            for (size_t j = 0; j < inputs.row_count_; j++) {
                weights_[j].grad_ += out_[i].grad_ 
                    * inputs[j * inputs.col_count_ + i].value_; 
            }
        }

        // average derivatives and apply to value
        bias_.value_ -= bias_.grad_ * learning_rate_ *
                        (1 / static_cast<double>(out_.row_count_));
        for (auto& weight_ : weights_.elements_) {
            weight_.value_ -= weight_.grad_ * learning_rate_ *
                              (1 / static_cast<double>(inputs.col_count_));
        }
    }

    void BackProp(std::vector<Neuron>& input) {
        // compute derivatives
        for (size_t i = 0; i < out_.size(); i++) {
            bias_.grad_ += out_[i].grad_;
            for (size_t j = 0; j < input.size(); j++) {
                weights_[j].grad_ += out_[i].grad_ ;
            }
        }

        // average derivatives
        bias_.value_ -= bias_.grad_ * learning_rate_ *
                        (1 / static_cast<double>(out_.row_count_));
        for (auto& weight_ : weights_.elements_) {
            weight_.value_ -= weight_.grad_ * learning_rate_ *
                              (1 / static_cast<double>(input[0].out_.col_count_));
        }
    }

    void ZeroGrad() {
        for (auto& element_ : out_.elements_) element_.grad_ = 0;
        bias_.grad_ = 0;
        for (auto& element_ : weights_.elements_) element_.grad_ = 0;
    }
};

#endif // !NEURON_H
