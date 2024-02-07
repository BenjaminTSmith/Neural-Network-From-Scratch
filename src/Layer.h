#ifndef LAYER_H
#define LAYER_H

#include "Neuron.h"
#include <algorithm>
#include <iterator>

template<typename T>
class Layer {
public:
    std::vector<T> neurons;

    Layer(int size, int nins) {
        neurons.resize(size);
        std::fill(neurons.begin(), neurons.end(), T(nins));
    }

    Layer(int size) {
        neurons.resize(size);
        std::fill(neurons.begin(), neurons.end(), T(0));
    }


    void forwardPass(const std::vector<Node>& inputs) {
        for (auto& neuron: neurons) { neuron.forwardPass(inputs); }
    }

    void forwardPass(const std::vector<double>& inputs) {
        for (auto& neuron: neurons) { neuron.forwardPass(inputs); }
    }

    void forwardPass(const Layer<Neuron>& inputs) {
        for (auto& neuron: neurons) {
            neuron.forwardPass(inputs.neurons);
        }
    }

    void forwardPass(const Layer<double>& inputs) {
        for (auto& neuron: neurons) {
            neuron.forwardPass(inputs.neurons);
        }
    }

    void backProp(std::vector<Node>& inputs) {
        for (auto& neuron: neurons) { neuron.backProp(inputs); }
    }

    void backProp(std::vector<double>& inputs) {
        for (auto& neuron: neurons) { neuron.backProp(inputs); }
    }

    void backProp(Layer<Neuron>& inputs) {
        for (auto& neuron: neurons) {
            neuron.backProp(inputs.neurons);
        }
    }

    void backProp(Layer<double>& inputs) {
        for (auto& neuron: neurons) {
            neuron.backProp(inputs.neurons);
        }
    }

    void zeroGrad() {
        for (auto& neuron: neurons) { neuron.zeroGrad(); }
    }

    void forwardProp() {
        for (auto& neuron: neurons) { neuron.forwardProp(); }
    }

};


#endif // !LAYER_H
