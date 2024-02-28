#ifndef NEURON_H
#define NEURON_H

#include <functional>
#include <iostream>

#include "eigen3/Eigen/Eigen"
#include "Activations.h"

namespace nn {

class Neuron {
public:
    Matrix activated_out_;
    Matrix out_;

    ColVector weights_;
    double bias_;

    double learning_rate_ = 1;

    std::function<Matrix(Matrix)> activation_function_ = Identity;
    std::function<Matrix(Matrix)> activation_derivative_ = d_Identity;

    Neuron(int size) : weights_(size), bias_(0) {
        weights_.setRandom();
    }

    Neuron() {}

    void set_learning_rate(double lr) { learning_rate_ = lr; }

    // Matrix rows are inputs. Matrix cols need to match nins
    // to layer and to neurons.
    Matrix Activate(const Matrix& inputs) {
        out_ = (inputs * weights_).array() + bias_;
        out_.transposeInPlace();
        activated_out_ = activation_function_(out_);
        return activated_out_;
    }

    void ComputeGradients(const Matrix& inputs, 
                                std::vector<Neuron>& prev_neurons) 
    {
        out_ = activation_derivative_(out_) * activated_out_;
        bias_ -= learning_rate_ * out_.rowwise().mean()[0];
        weights_.array() -= inputs.colwise().mean().array() *
            out_.array() * learning_rate_;
        for (auto& neuron : prev_neurons) {
            neuron.activated_out_.array() += weights_.array()
                * out_.array();
        }
    }

};

}

#endif  // !NEURON_H
