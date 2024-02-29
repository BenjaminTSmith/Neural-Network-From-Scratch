#ifndef LAYER_H
#define LAYER_H

#include "Activations.h"
#include "Neuron.h"
#include <iostream>

namespace nn {

class Layer {
public:
    std::vector<nn::Neuron> neurons_;

    Matrix out_;
    Matrix activated_out_;

    std::function<Matrix(Matrix)> activation_function_ = Identity;
    std::function<Matrix(Matrix)> activation_derivative_ = d_Identity;

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

    void SetDeltaOne() {
        for (auto& neuron : neurons_) {
            neuron.activated_out_ = Matrix::Ones(neuron.activated_out_.rows(),
                                                 neuron.activated_out_.cols());
        }
    }

    void PrintOutput() {
        for (const auto& neuron : neurons_) {
            std::cout << neuron.out_ << std::endl;
        }
    }

    // Matrix rows are inputs. Matrix cols need to match nins
    // to layer and to neurons.
    Matrix ForwardPass(const Matrix& inputs) {
        out_ = Matrix(neurons_.size(), inputs.rows());
        for (int i = 0; i < neurons_.size(); ++i) { 
            out_.row(i) = neurons_[i].Activate(inputs); 
        }
        activated_out_ = activation_function_(out_);
        return activated_out_;
    }

    Matrix ForwardPass(Layer& prev_layer) {
        return ForwardPass(prev_layer.out_);
    }

    void BackProp(Layer& prev_layer) {
        for (auto& neuron: neurons_) {
            neuron.ComputeGradients(prev_layer.activated_out_,
                                    prev_layer.neurons_);
        }
    }

};

}

#endif // !LAYER_H
