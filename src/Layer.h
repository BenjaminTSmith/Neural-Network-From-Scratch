#ifndef LAYER_H
#define LAYER_H

#include "Neuron.h"
#include <iostream>

namespace nn {

class Layer {
public:
    std::vector<nn::Neuron> neurons_;
    Matrix out_;
    ColVector deltas_;

    Layer(int size, int nins) {
        neurons_.resize(size);
        for (auto& neuron: neurons_) neuron = nn::Neuron(nins);
        deltas_ = Matrix::Zero(size, 1);
    }

    Layer(int size) {
        neurons_.resize(size);
        for (auto& neuron: neurons_) neuron = nn::Neuron(0);
    }

    [[nodiscard]]
    nn::Neuron& operator[](int index) { return neurons_[index]; }

    void set_learning_rate(double lr) {
        for (auto& neuron: neurons_) { neuron.set_learning_rate(lr); }
    }

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

    void ForwardPass(Layer& prev_layer) {
        ForwardPass(prev_layer.out_);
    }

    void BackProp(Matrix& inputs, Matrix& input_grads) {
        for (size_t i = 0; i < neurons_.size(); ++i) { 
            neurons_[i].BackProp(inputs, input_grads, deltas_[i]); 
        }
    }

    void BackProp(Layer& prev_layer) {
        // Todo: deltas and weights should all be extrapolated to Layer
        // class instead of Neuron. Layer tells neuron what to differentiate
        // with respect to instead of the Neuron having to keep track of it.
        // This will make it so that vectors of neurons don't have to
        // continuously be converted to and from matrices and vectors.
    }

};

}

#endif // !LAYER_H
