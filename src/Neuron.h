#ifndef NEURON_H
#define NEURON_H

#include <iostream>
#include "eigen3/Eigen/Eigen"
#include "Node.h"

namespace nn {

typedef Eigen::MatrixX<Node> Matrix;
typedef Eigen::VectorX<Node> Vector;
typedef Eigen::RowVectorX<Node> RowVector;

class Neuron {
public:
    std::vector<Node> weights_;
    Node bias_;
    Node out_;
    
    Neuron(int nins) : weights_(nins) {}

    Node ForwardPass(const std::vector<Node>& input) {
        if (weights_.size() != input.size()) 
            std::cerr << "input.size() != weights_.size()" << std::endl;
        
        for (int i = 0; i < input.size(); i++) 
            out_ += weights_[i] * input[i];
        

        out_ += bias_;
        return out_;
    }

    void BackProp() {
        out_.ComputeGradients();
    }

    operator Node() const { return out_; }

};

}

#endif  // !NEURON_H
