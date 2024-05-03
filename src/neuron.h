#ifndef NEURON_H
#define NEURON_H

#include "dval.h"
#include "matrix.h"

struct Neuron {
    Matrix<Dval> weights_;
    Dval bias_;
    Matrix<Dval> out_;

    Neuron(int nins) : weights_(1, nins), bias_(0) {}

    Neuron() {}

    Matrix<Dval> ForwardPass(const Matrix<Dval>& inputs) {
        // weights_ is a row vector, input needs to be a column vector or
        // a matrix with columns as inputs
        out_ = weights_ * inputs + bias_;
        out_ = out_.Transpose(); // ReLU and Transpose
        return out_;
    }

    Matrix<Dval> ForwardPass(const std::vector<Neuron>& inputs) {
        Matrix<Dval> input_matrix(inputs.size(), inputs[0].out_.size());
        std::vector<Dval> input_vector;
        for (auto& input : inputs) {
            for (auto& element_ : input.out_.elements_) {
                input_vector.push_back(element_);
            }
        }
        input_matrix.SetElements(input_vector);

        out_ = weights_ * input_matrix + bias_;
        out_ = out_.Transpose();
        return out_;
    }

    void BackProp(const Matrix<Dval>& inputs, double alpha) {
        for (size_t i = 0; i < out_.size(); i++) {
            bias_.grad_ += out_[i].grad_;
            for (size_t j = 0; j < inputs.row_count_; j++) {
                if (out_[i].value_ > 0) {
                    weights_[j].grad_ += out_[i].grad_ 
                        * inputs[j * inputs.col_count_ + i].value_; 
                }
            }
        }

        // average derivatives and apply to value
        bias_.value_ -= bias_.grad_ * alpha *
                        (1 / static_cast<double>(out_.row_count_));
        for (auto& weight_ : weights_.elements_) {
            weight_.value_ -= weight_.grad_ * alpha *
                              (1 / static_cast<double>(out_.row_count_));
        }
    }

    void BackProp(std::vector<Neuron>& input, double alpha) {
        for (size_t i = 0; i < out_.size(); i++) {
            bias_.grad_ += out_[i].grad_;
            for (size_t j = 0; j < input.size(); j++) {
                if (out_[i].value_ > 0) {
                    weights_[j].grad_ += out_[i].grad_ * input[j].out_[i].value_;
                    input[j].out_[i].grad_ += out_[i].grad_ * weights_[j].value_;
                }
            }
        }

        bias_.value_ -= bias_.grad_ * alpha *
                        (1 / static_cast<double>(out_.row_count_));
        for (auto& weight_ : weights_.elements_) {
            weight_.value_ -= weight_.grad_ * alpha *
                              (1 / static_cast<double>(out_.row_count_));
        }
    }

    void Activate() {
        out_ = out_.Max(0);
    }

    void ZeroGrad() {
        for (auto& element_ : out_.elements_) element_.grad_ = 0;
        bias_.grad_ = 0;
        for (auto& element_ : weights_.elements_) element_.grad_ = 0;
    }
};

#endif // !NEURON_H
