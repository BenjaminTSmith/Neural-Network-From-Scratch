/*#ifndef LAYER_H
#define LAYER_H

#include "Matrix.h"
#include "Neuron.h"
#include <iostream>
#include <cmath>

class Layer {
public:
    std::vector<Neuron> neurons_;
    double loss_ = 1000;

    Layer(int size, int nins) {
        neurons_.resize(size);
        std::fill(neurons_.begin(), neurons_.end(), Neuron(nins));
    }

    Layer(int size) {
        neurons_.resize(size);
        std::fill(neurons_.begin(), neurons_.end(), Neuron(0));
    }

    void SetInputLayer(const std::vector<double>& inputs) {
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

    [[nodiscard]]  std::vector<double> GetOutput() {
        std::vector<double> output;
        for (const auto& neuron: neurons_) {
            output.push_back(neuron.out_.value_);
        }
        return output;
    }

    void ForwardPass(const Layer& inputs) {
        for (auto& neuron: neurons_) { neuron.ForwardPass(inputs.neurons_); }
    }

    void BackProp(Layer& inputs) {
        for (auto& neuron: neurons_) { neuron.BackProp(inputs.neurons_); }
    }

    void ForwardProp() {
        for (auto& neuron: neurons_) { neuron.ForwardProp(); }
    }

    void ZeroGrad() {
        for (auto& neuron: neurons_) { neuron.ZeroGrad(); }
    }

    void AverageGrad(int batch_size) {
        for (auto& neuron: neurons_) { neuron.AverageGrad(batch_size); }
    }

    void ReLU() {
        for (auto& neuron: neurons_) {
            neuron.out_ = neuron.out_ > 0 ? neuron.out_ : 0;
            neuron.activation_grad_ *= neuron.sum_ > 0 ? 1 : 0;
        }
    }

    void LeakyReLU() {
        for (auto& neuron: neurons_) {
            neuron.out_ = neuron.out_ > 0 ? neuron.out_ : neuron.out_ * 0.1;
            neuron.activation_grad_ *= neuron.sum_ > 0 ? 1 : 0.1;
        }
    }

    void SoftMax() {
        double max = neurons_[0].out_.value_;
        for (auto& neuron: neurons_) {
            if (neuron.out_.value_ > max) { max = neuron.out_.value_; }
        }

        double sum = 0;
        for (auto& neuron: neurons_) {
            sum += std::exp(neuron.out_.value_ - max);
        }

        for (auto& neuron: neurons_) {
            neuron.out_.value_ = std::exp(neuron.out_.value_ - max) / sum;
            neuron.activation_grad_ *= neuron.out_.value_ * (1 - neuron.out_.value_);
        }
    }

    void MSE(const std::vector<double>& ground_truth) {
        loss_ = 0;
        for (int i = 0; i < neurons_.size(); ++i) {
            double temp = neurons_[i].out_.value_ - ground_truth[i];
            loss_ += temp * temp;

            neurons_[i].activation_grad_ *= 2.0 / ground_truth.size() * temp;
            neurons_[i].out_.grad_ = 1;
        }
        loss_ /= neurons_.size();
    }

    void SparseCategoricalCrossEntropy(const std::vector<double>& one_hot_vector) {
        loss_ = 0;
        auto output = GetOutput();
        Clip(output, 1e-3, 1 - 1e-3);
        double test = 1 - 1e-7;
        for (int i = 0; i < one_hot_vector.size(); ++i) {
            loss_ += -one_hot_vector[i] * std::log(output[i]);

            neurons_[i].activation_grad_ *= (output[i] - one_hot_vector[i]);
            neurons_[i].out_.grad_ = 1;
        }
        loss_ /= one_hot_vector.size();
    }
};


#endif // !LAYER_H*/
