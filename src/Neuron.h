#ifndef NEURON_H
#define NEURON_H

#include "Node.h"
#include <vector>

class Neuron {
public:
    std::vector<Node> weights_;
    Node out_;
    Node bias_;
    Node sum_;
    double learning_rate = 0.01;

    Neuron(int size) {
        weights_.resize(size);
    }   

    Neuron() {}

    void ForwardPass(const std::vector<Neuron>& inputs) {
        sum_.value_ = 0;
        for (int i = 0; i < inputs.size(); i++) {
            sum_ += inputs[i].out_.value_ * weights_[i].value_;
        }
        sum_ += bias_;
    }

    void BackProp(std::vector<Neuron>& inputs) {
        double temp = sum_.value_ > 0 ? 1 : 0;
        sum_.grad_ = temp * out_.grad_;
        bias_.grad_ = sum_.grad_;

        for (int i = 0; i < inputs.size(); i++) {
            inputs[i].out_.grad_ += sum_.grad_ * weights_[i].value_;
            weights_[i].grad_ = sum_.grad_ * inputs[i].out_.value_;
        }
    }

    void ForwardProp() {
        for (auto& weight: weights_) {
            weight.value_ -= weight.grad_ * learning_rate;
        }
        bias_.value_ -= bias_.grad_ * learning_rate;
    }

    void ZeroGrad() {
        sum_.grad_ = 0;
        out_.grad_ = 0;
        for (auto& weight: weights_) {
            weight.grad_ = 0;
        }
        bias_.grad_ = 0;
    }
};

#endif // !NEURON_H
