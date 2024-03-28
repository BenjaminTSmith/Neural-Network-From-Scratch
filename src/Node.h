#ifndef NODE_H
#define NODE_H

#include <vector>
// #include "eigen3/Eigen/Eigen"

namespace DAG {

enum class Op {
    ADD,
    MULTIPLY,
    SUBTRACT,
    DIVIDE,
    ReLU
};

struct Node {

    double value_ = 0;
    double grad_ = 0;
    
    Op op_;
    std::vector<Node*> children_;

    Node() {}

    Node(double value, double grad, std::vector<Node*> children)
        : value_(value),
          grad_(grad),
          children_(children) {}

    Node(double value, Op op,
         std::vector<Node*> children)
        : value_(value),
          op_(op),
          children_(children) {}

    Node(double value) : value_(value) {}
    
    Node(Node &&) = default;

    Node(const Node& other)
        : value_(other.value_),
          grad_(other.grad_),
          op_(other.op_),
          children_(other.children_) {}

    Node& operator=(Node &&) = default;

    Node& operator=(double value) {
        value_ = value;
        return *this;
    }

    Node& operator=(const Node& other) {
        value_ = other.value_;
        grad_ = other.grad_;
        children_ = other.children_;
        return *this;
    }

    Node operator+(Node& other) {
        return Node(other.value_ + value_,
                    Op::ADD,
                    { &other, this });
    }

    Node operator*(Node& other) {
        return Node(other.value_ * value_,
                    Op::MULTIPLY,
                    { &other, this });
    }

    Node operator/(Node& other) {
        return Node(value_ / other.value_,
                    Op::DIVIDE,
                    { this, &other });
    }

    Node operator-(Node& other) {
        return Node(value_ - other.value_,
                    Op::SUBTRACT,
                    { this, &other });
    }

    Node ReLU() {
        return *new Node(std::max(0.0, this->value_),
                         Op::ReLU,
                         { this });
    }

    operator double() const { return value_; }

    void ComputeGradients() const {
        switch (op_) {
            case Op::ADD: 
                for (auto& child : children_) {
                    child->grad_ += grad_;
                }
                break;
            case Op::MULTIPLY:
                children_[0]->grad_ += children_[1]->value_ * grad_;
                children_[1]->grad_ += children_[0]->value_ * grad_;
                break;
            case Op::ReLU:
                children_[0]->grad_ += (value_ > 0 ? 1 : 0) * grad_;
                break;
            default:
                break;
        }
    }

    void ZeroGrad() {
        grad_ = 0;
        for (auto& child : children_) child->ZeroGrad();
    }

    void AverageGrad(int batch_size) {
        grad_ /= batch_size; 
        for (auto& child : children_) child->AverageGrad(batch_size);
    }

};

}

/*namespace Eigen {

template<> struct NumTraits<nn::Node>
    : NumTraits<double> {
    typedef nn::Node Real;
    typedef nn::Node NonInteger;
    typedef nn::Node Nested;
    enum {
        IsComplex = 0,
        IsInteger = 0,
        IsSigned = 1,
        RequireInitialization = 1,
        ReadCost = 1,
        AddCost = 1,
        MulCost = 1
    };
};

}*/

#endif // !NODE_H
