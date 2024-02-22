#ifndef LAYER_H
#define LAYER_H

#include "Neuron.h"
#include <iostream>

class Layer {
public:
    std::vector<Neuron> neurons_;
    double loss_ = 0;

    Layer(int size, int nins) {
        neurons_.resize(size);
        std::fill(neurons_.begin(), neurons_.end(), Neuron(nins));
    }

    Layer(int size) {
        neurons_.resize(size);
        std::fill(neurons_.begin(), neurons_.end(), Neuron(0));
    }

    void SetInputs(const std::vector<double>& inputs) {
        for (int i = 0; i < inputs.size(); ++i) { 
            neurons_[i].out_.value_ = inputs[i]; 
        }
    }

    Neuron& operator[](int index) { return neurons_[index]; }

    void PrintOutput() {
        for (const auto& neuron: neurons_) {
            std::cout << neuron.out_.value_ << std::endl;
        }
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
            neuron.out_.value_ = neuron.sum_ > 0 ? neuron.sum_.value_ : 0;
            neuron.out_.grad_ = neuron.sum_ > 0 ? 1 : 0;
        }
    }

    void MSE(const std::vector<double> ground_truth) {
        for (int i = 0; i < neurons_.size(); ++i) {
            neurons_[i].out_.value_ = (neurons_[i].out_.value_ - ground_truth[i]) *
                               (neurons_[i].out_.value_ - ground_truth[i]);
            loss_ += neurons_[i].out_.value_;

            neurons_[i].out_.grad_ = 2 * (neurons_[i].out_.value_ - 
                                            ground_truth[i]);
        }
        loss_ /= neurons_.size();
    }

    void SoftMax() {
        // todo
    }

};


#endif // !LAYER_H
