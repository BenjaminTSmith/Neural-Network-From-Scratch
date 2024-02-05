#ifndef LAYER_H
#define LAYER_H

#include "Neuron.h"
#include <algorithm>

template<typename T>
class Layer {
public:
    std::vector<T> neurons;

    Layer(int size, int nins) {
        neurons.resize(size);
        std::fill(neurons.begin(), neurons.end(), Neuron(nins));
    }

    Layer(int size) {
        neurons.resize(size);
        std::fill(neurons.begin(), neurons.end(), int(0));
    }

    void forwardPass(const std::vector<Node>& inputs) {
        for (auto& neuron: neurons) { neuron.forwardPass(inputs); }
    }

    void forwardPass(const std::vector<int>& inputs) {
        for (auto& neuron: neurons) { neuron.forwardPass(inputs); }
    }

    void forwardPass(const Layer& inputs) {
        std::vector<Node> prevLayer;
        for (auto& input: inputs.neurons) {
            prevLayer.push_back(input.out);
        }
        for (auto& neuron: neurons) {
            neuron.forwardPass(prevLayer);
        }
    }

    void backProp(std::vector<Node>& inputs) {
        for (auto& neuron: neurons) { neuron.backProp(inputs); }
    }

    void backProp(std::vector<int>& inputs) {
        for (auto& neuron: neurons) { neuron.backProp(inputs); }
    }

    void zeroGrad() {
        for (auto& neuron: neurons) { neuron.zeroGrad(); }
    }

    void forwardProp() {
        for (auto& neuron: neurons) { neuron.forwardProp(); }
    }

};


#endif // !LAYER_H
