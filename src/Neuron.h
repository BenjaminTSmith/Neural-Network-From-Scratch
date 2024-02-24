#ifndef NEURON_H
#define NEURON_H

#include "eigen3/Eigen/Eigen"
#include "iostream"

typedef Eigen::VectorXd ColVector;
typedef Eigen::RowVectorXd RowVector;
typedef Eigen::MatrixXd Matrix;

class Neuron {
public:
    ColVector weights_;
    double out_;
    Matrix matrix_out_;
    double bias_;
    double learning_rate_ = 1;

    Neuron(int size) : weights_(size), bias_(0) {
        weights_.setRandom();
        // std::cout << weights_ << std::endl;
    }

    double ForwardPass(const RowVector& inputs) {
        return out_ = inputs * weights_ + bias_;
    }

    Matrix ForwardPass(const Matrix& inputs) {
        matrix_out_ = inputs * weights_;
        return matrix_out_.array() += bias_;
    }

    void BackProp(std::vector<Neuron>& inputs) {}

    void ForwardProp() {}

    void ZeroGrad() {}
};

#endif  // !NEURON_H
