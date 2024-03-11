#ifndef NODE_H
#define NODE_H

#include <memory>
#include <iostream>
#include "eigen3/Eigen/Eigen"

using std::shared_ptr;

namespace nn {

enum class Operation {
    add,
    multiply,
    ReLU
};

class Node {
public:
    double value_ = 0;
    double grad_ = 0;
    Operation op_;
    std::vector<Node*> children_;

    Node(double value, Operation op,
         std::vector<Node*> children)
        : value_(value),
          op_(op),
          children_(children) {}
    Node(double value) : value_(value) {}
    Node() { SetRandom(); }
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

    Node& operator+(Node& other) {
        return *new Node(other.value_ + value_,
                         Operation::add,
                         { &other, this });
    }

    void operator+=(Node& other) {
        *this =  *new Node(*this) + other;
    }

    Node& operator*(Node& other) {
        return *new Node(other.value_ + value_,
                         Operation::add,
                         { &other, this });
    }

    void operator*=(Node& other) {
        *this = *new Node(*this) * other;
    }

    Node ReLU() {
         /*return *std::make_shared<Node>(Node(value_ > 0 ? value_ : 0,
                                            Operation::ReLU,
                                            { this }));*/
        return 1;
    }

    bool operator==(const Node& other) const {
        return value_ == other.value_;
    }

    void SetRandom() {
        value_ = static_cast<double>(rand()) / RAND_MAX * 2.0 - 1.0;
    }

    operator double() const { return value_; }

    void ComputeGradients() {
        switch (op_) {
            case Operation::add: 
                for (auto& child : children_) {
                    child->grad_ += grad_;
                }
                break;
            case Operation::multiply:
                children_[0]->grad_ += children_[1]->value_ * grad_;
                children_[1]->grad_ += children_[0]->value_ * grad_;
                break;
            case Operation::ReLU:
                children_[0]->grad_ += (value_ > 0 ? 1 : 0) * grad_;
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

namespace Eigen {

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

}

#endif // !NODE_H
