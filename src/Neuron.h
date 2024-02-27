#ifndef NEURON_H
#define NEURON_H

#include "eigen3/Eigen/Eigen"

namespace nn {

typedef Eigen::VectorXd ColVector;
typedef Eigen::RowVectorXd RowVector;
typedef Eigen::MatrixXd Matrix;

class Neuron {
public:
    Matrix out_;
    ColVector weights_;
    ColVector weight_grads_;
    double bias_;
    double bias_grad_;
    double learning_rate_ = 1;

    Neuron(int size) : weights_(size), bias_(0) {
        weights_.setRandom();
    }

    Neuron() {}

    void set_learning_rate(double lr) { learning_rate_ = lr; }

    Matrix ForwardPass(const Matrix& inputs) {
        out_ = (inputs * weights_).array() + bias_;
        out_.transposeInPlace();
        return out_;
    }

    void BackProp(const Matrix& inputs, Matrix& input_grads, double delta) {
        bias_grad_ = delta * learning_rate_;
        weight_grads_ = inputs.colwise().mean() * delta;
        input_grads += weights_.transpose() * delta;
    }
};

}

#endif  // !NEURON_H
