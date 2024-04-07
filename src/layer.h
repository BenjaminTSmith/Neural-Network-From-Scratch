#ifndef LAYER_H
#define LAYER_H 

#include "neuron.h"

class Layer {
    std::vector<Neuron> neurons_;
    Matrix<Dval> out_;

    Layer(int nneurons, int nins) : neurons_(nneurons) {
        for (auto& neuron_ : neurons_) neuron_ = Neuron(nins);
    }

    void ForwardProp(const Layer& in) {}

    void ForwardProp(const Matrix<Dval>& in) {
        for (auto& neuron_ : neurons_) neuron_.ForwardProp(in);
    }
};

#endif // !LAYER_H
