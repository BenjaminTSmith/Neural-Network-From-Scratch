#ifndef NEURON_H
#define NEURON_H

#include <functional>
#include "eigen3/Eigen/Eigen"
#include "Activations.h"

namespace nn {

typedef Eigen::VectorXd ColVector;
typedef Eigen::RowVectorXd RowVector;
typedef Eigen::MatrixXd Matrix;

class Neuron {
public:
    Matrix activated_out_;
    Matrix out_;

    ColVector weights_;
    double bias_;

    double learning_rate_ = 1;
    std::function<Matrix(Matrix)> activation_function_;

    Neuron(int size) : weights_(size), bias_(0) {
        weights_.setRandom();
        activation_function_ = ReLU;
    }

    Neuron() {}

    void set_learning_rate(double lr) { learning_rate_ = lr; }

    Matrix ForwardPass(const Matrix& inputs) {
        out_ = (inputs * weights_).array() + bias_;
        out_.transposeInPlace();
        activated_out_ = activation_function_(out_);
        return activated_out_;
    }

    void ComputeInputGradients(const Matrix& inputs, double delta) {
        bias_ -= delta * learning_rate_;
        weights_ -= inputs.colwise().mean() * delta * learning_rate_;
    }

};

}

#endif  // !NEURON_H
