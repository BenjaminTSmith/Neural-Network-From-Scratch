#ifndef LAYER_H
#define LAYER_H

#include "Neuron.h"
#include <iostream>

namespace nn {

class Layer {
public:
    std::vector<nn::Neuron> neurons_;
    Matrix out_;

    Layer(int size, int nins) {
        neurons_.resize(size);
        for (auto& neuron: neurons_) neuron = nn::Neuron(nins);
    }

    Layer(int size) {
        neurons_.resize(size);
        for (auto& neuron: neurons_) neuron = nn::Neuron(0);
    }

    [[nodiscard]]
    nn::Neuron& operator[](int index) { return neurons_[index]; }

    void PrintOutput() {
        for (const auto& neuron: neurons_) {
            std::cout << neuron.out_ << std::endl;
        }
    }

    // Matrix rows are inputs. Matrix cols need to match nins
    // to layer and to neurons.
    Matrix ForwardPass(const Matrix& inputs) {
        out_ = Matrix(neurons_.size(), inputs.rows());
        for (int i = 0; i < neurons_.size(); ++i) {
            out_.row(i) = neurons_[i].ForwardPass(inputs);
        }
        return out_;
    }

};

}

#endif // !LAYER_H
