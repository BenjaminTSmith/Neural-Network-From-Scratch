#ifndef LAYER_H
#define LAYER_H

#include "Neuron.h"
#include <algorithm>
#include <iterator>

template<typename T>
class Layer {
public:
    std::vector<T> neurons_;

    Layer(int size, int nins) {
        neurons_.resize(size);
        std::fill(neurons_.begin(), neurons_.end(), T(nins));
    }

    Layer(int size) {
        neurons_.resize(size);
        std::fill(neurons_.begin(), neurons_.end(), T(0));
    }


    void ForwardPass(const std::vector<Node>& inputs) {
        for (auto& neuron: neurons_) { neuron.ForwardPass(inputs); }
    }

    void ForwardPass(const std::vector<double>& inputs) {
        for (auto& neuron: neurons_) { neuron.ForwardPass(inputs); }
    }

    void ForwardPass(const Layer<Neuron>& inputs) {
        for (auto& neuron: neurons_) {
            neuron.ForwardPass(inputs.neurons_);
        }
    }

    void ForwardPass(const Layer<double>& inputs) {
        for (auto& neuron: neurons_) {
            neuron.ForwardPass(inputs.neurons_);
        }
    }

    void OutPass(const Layer<Neuron>& inputs) {
        for (auto& neuron: neurons_) {
            neuron.OutPass(inputs.neurons_);
        }
    }

    void BackProp(std::vector<Node>& inputs) {
        for (auto& neuron: neurons_) { neuron.BackProp(inputs); }
    }

    void OutProp(std::vector<Node>& inputs) {
        for (auto& neuron: neurons_) { neuron.OutProp(inputs); }
    }

    void BackProp(std::vector<double>& inputs) {
        for (auto& neuron: neurons_) { neuron.BackProp(inputs); }
    }

    void BackProp(Layer<Neuron>& inputs) {
        for (auto& neuron: neurons_) {
            neuron.BackProp(inputs.neurons_);
        }
    }

    void BackProp(Layer<double>& inputs) {
        for (auto& neuron: neurons_) {
            neuron.BackProp(inputs.neurons_);
        }
    }

    void OutProp(Layer<Neuron>& inputs) {
        for (auto& neuron: neurons_) {
            neuron.OutProp(inputs.neurons_);
        }
    }

    void ZeroGrad() {
        for (auto& neuron: neurons_) { neuron.ZeroGrad(); }
    }

    void ForwardProp() {
        for (auto& neuron: neurons_) { neuron.ForwardProp(); }
    }

};


#endif // !LAYER_H
