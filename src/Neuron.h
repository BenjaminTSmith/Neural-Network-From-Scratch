#ifndef NEURON_H
#define NEURON_H

#include "eigen3/Eigen/Eigen"

typedef Eigen::VectorXd ColVector;
typedef Eigen::RowVectorXd RowVector;
typedef Eigen::MatrixXd Matrix;

namespace nn {

class Neuron {
public:
    Matrix out_;
    ColVector weights_;
    double bias_;
    double learning_rate_ = 1;
    double delta_;

    Neuron(int size) : weights_(size), bias_(0), delta_(1) {
        weights_.setRandom();
    }

    Neuron() {}

    void set_out_grad(double grad) { delta_ = grad; }
    void set_learning_rate(double lr) { learning_rate_ = lr; }

    Matrix ForwardPass(const Matrix& inputs) {
        out_ = (inputs * weights_).array() + bias_;
        out_.transposeInPlace();
        return out_;
    }

    void BackProp(const Matrix& inputs) {
        bias_ -= delta_ * learning_rate_;
        weights_ -= inputs.colwise().mean() * learning_rate_ * delta_;
    }
};

}

#endif  // !NEURON_H
