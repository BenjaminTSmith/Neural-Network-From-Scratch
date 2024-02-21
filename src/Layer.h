#ifndef LAYER_H
#define LAYER_H

#include "Neuron.h"

class Layer {
public:
    std::vector<Neuron> neurons_;

    Layer(int size, int nins) {
        neurons_.resize(size);
        std::fill(neurons_.begin(), neurons_.end(), Neuron(nins));
    }

    Layer(int size) {
        neurons_.resize(size);
        std::fill(neurons_.begin(), neurons_.end(), Neuron(0));
    }

    void ForwardPass(const Layer& inputs) {
        for (auto& neuron: neurons_) {
            neuron.ForwardPass(inputs.neurons_);
        }
    }

    void BackProp(Layer& inputs) {
        for (auto& neuron: neurons_) {
            neuron.BackProp(inputs.neurons_);
        }
    }

    void ForwardProp() {
        for (auto& neuron: neurons_) { neuron.ForwardProp(); }
    }

    void ZeroGrad() {
        for (auto& neuron: neurons_) { neuron.ZeroGrad(); }
    }

    void ReLU() {
        for (auto& neuron: neurons_) {
            neuron.out_ = neuron.sum_ > 0 ? neuron.sum_ : 0;
        }
    }

};


#endif // !LAYER_H
