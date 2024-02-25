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
    double out_grad_;

    Neuron(int size) : weights_(size), bias_(1), out_grad_(0) {
        weights_.setRandom();
    }

    void set_out_grad(double grad) { out_grad_ = grad; }

    Matrix ForwardPass(const Matrix& inputs) {
        out_ = inputs * weights_;
        out_ = out_.array() + bias_;
        out_.transposeInPlace();
        return out_;
    }

    void BackProp(const Matrix& inputs) {

    }
};

}

#endif  // !NEURON_H
