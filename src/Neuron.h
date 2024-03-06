#ifndef NEURON_H
#define NEURON_H

#include "eigen3/Eigen/Eigen"
#include "Node.h"

namespace nn {

typedef Eigen::MatrixXd Matrix;

class Neuron {
public:
    Node out_;
    Node weights_;
    Matrix activation_delta_;
    double bias_;
    
    double learning_rate_ = 1;

    Neuron(int size) : weights_(size, 1), bias_(0) {
        weights_.value_.setRandom();
    }

    Neuron() {}

    void set_learning_rate(double lr) { learning_rate_ = lr; }

    // Matrix rows are inputs. Matrix cols need to match nins
    // to layer and to neurons.
    void Activate(const Matrix& inputs) {
        out_.value_ = (inputs * weights_.value_).array() + bias_;
    }

    void ZeroGrad() {
        out_.ZeroGrad();
        weights_.ZeroGrad();
    }

    void BackProp() {
        weights_.value_ -= weights_.grad_ * learning_rate_;
    }

    void ComputeGradients(const Matrix& inputs, 
                                std::vector<Neuron>& prev_neurons) {
        // not sure if this is right yet
        out_.grad_.array() *= activation_delta_.array(); 
        bias_ -= learning_rate_ * out_.mean();
        for (auto& neuron : prev_neurons) {
            // not sure if this is right yet
            neuron.out_.grad_.array() += weights_.value_.array()
                * out_.grad_.array();
        }
        weights_.grad_.array() += inputs.colwise().mean().array() * 
            out_.grad_.mean();
    }
    
    void ReLU() {
        out_.value_ = out_.value_.cwiseMax(0);
        activation_delta_ = out_.value_.unaryExpr([](double num) {
            return num > 0 ? 1 : 0; 
        });
    }

    void Softmax(const std::vector<Neuron>& inputs) {
        
    }

};

}

#endif  // !NEURON_H
