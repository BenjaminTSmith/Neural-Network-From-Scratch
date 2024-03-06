#ifndef LAYER_H
#define LAYER_H

#include "Neuron.h"
#include <iostream>
#include <functional>

namespace nn {

class Layer {
public:
    std::vector<nn::Neuron> neurons_;

    Matrix out_;

    std::function<Matrix(Matrix)> activation_function_;

    Layer(int size, int nins) {
        neurons_.resize(size);
        for (auto& neuron : neurons_) neuron = nn::Neuron(nins);
    }

    Layer(int size) {
        neurons_.resize(size);
        for (auto& neuron : neurons_) neuron = nn::Neuron(0);
    }

    [[nodiscard]]
    nn::Neuron& operator[](int index) { return neurons_[index]; }

    void set_learning_rate(double lr) {
        for (auto& neuron : neurons_) { neuron.set_learning_rate(lr); }
    }

    void ZeroGrad() {
        for (auto& neuron : neurons_) {
            // to do.
            neuron.out_.delta_.fill(0);
        }
    }

    void PrintOutput() {
        std::cout << out_ << std::endl;
    }

    // Matrix rows are inputs. Matrix cols need to match nins
    // to layer and to neurons.
    Matrix ForwardPass(const Matrix& inputs) {
        out_ = Matrix(inputs.rows(), neurons_.size());
        for (int i = 0; i < neurons_.size(); ++i) { 
            out_.col(i) = neurons_[i].Activate(inputs); 
        }
        out_ = activation_function_(out_);
        return out_;
    }

    Matrix ForwardPass(Layer& prev_layer) {
        return ForwardPass(prev_layer.out_);
    }

    void BackProp(const Matrix& inputs) {
        for (auto& neuron: neurons_) {
            neuron.ComputeGradients(inputs);
        }
    }

    void BackProp(Layer& prev_layer) {
        prev_layer.ZeroGrad();
        for (auto& neuron: neurons_) {
            neuron.ComputeGradients(prev_layer.out_,
                                    prev_layer.neurons_);
        }
    }

};

}

#endif // !LAYER_H
