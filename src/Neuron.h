#ifndef NEURON_H
#define NEURON_H

#include "Node.h"
#include <vector>
#include <iostream>

class Neuron {
public:
    std::vector<Node> weights;
    Node out;
    Node bias;
    Node sum;

    Neuron() {}

    Neuron(size_t size) {
        weights.resize(size);
    }


    void forwardPass(const std::vector<Node>& inputs) {
        sum.value = 0;
        for (int i = 0; i < inputs.size(); i++) {
            sum += inputs[i] * weights[i];
        }
        sum += bias;

        out = sum.tanh();
    }

    void forwardPass(const std::vector<double>& inputs) {
        sum.value = 0;
        for (int i = 0; i < inputs.size(); i++) {
            sum += weights[i] * inputs[i];
        }
        sum += bias;

        out = sum.tanh();
    }

    void backProp(std::vector<Node>& inputs) {
        double temp = (std::exp(2 * sum.value) - 1) / (std::exp(2 * sum.value) + 1);
        sum.grad = (1 - temp * temp) * out.grad;
        bias.grad = sum.grad;

        for (int i = 0; i < inputs.size(); i++) {
            inputs[i].grad += sum.grad * weights[i].value;
            weights[i].grad = sum.grad * inputs[i].value;
        }
    }

    void backProp(std::vector<int>& inputs) {
        double temp = (std::exp(2 * sum.value) - 1) / (std::exp(2 * sum.value) + 1);
        sum.grad = (1 - temp * temp) * out.grad;
        bias.grad = sum.grad;

        for (int i = 0; i < inputs.size(); i++) {
            weights[i].grad = sum.grad * inputs[i];
        }
    }

    void forwardProp() {
        for (auto& weight: weights) {
            weight.value -= weight.grad;
        }
        bias.value -= bias.grad;
    }

    void zeroGrad() {
        out.grad = 0;
        for (auto& weight: weights) {
            weight.grad = 0;
        }
    }
};

#endif // !NEURON_H
