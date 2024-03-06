#ifndef NEURON_H
#define NEURON_H

#include "eigen3/Eigen/Eigen"
#include "Node.h"

namespace nn {

typedef Eigen::VectorXd ColVector;
typedef Eigen::RowVectorXd RowVector;
typedef Eigen::MatrixXd Matrix;

class Neuron {
public:
    Node out_;

    Node weights_;
    double bias_;

    double learning_rate_ = 1;

    Neuron(int size) : weights_(size, 1), bias_(0) {
        weights_.value_.setRandom();
    }

    Neuron() {}

    void set_learning_rate(double lr) { learning_rate_ = lr; }

    // Matrix rows are inputs. Matrix cols need to match nins
    // to layer and to neurons.
    Matrix Activate(const Matrix& inputs) {
        out_.value_ = (inputs * weights_.value_).array() + bias_;
        return out_.value_;
    }

    void ComputeGradients(const Matrix& inputs, 
                                std::vector<Neuron>& prev_neurons) {
        // add activation derivation
        // out_ = activation_derivative_(out_).array() * activated_out_.array();
        bias_ -= learning_rate_ * out_.mean();
        for (auto& neuron : prev_neurons) {
            neuron.out_.value_.array() += weights_.value_.array()
                * out_.value_.array();
        }
        weights_.value_.array() -= inputs.colwise().mean().array() *
            out_.mean() * learning_rate_;
    }

};

}

#endif  // !NEURON_H
