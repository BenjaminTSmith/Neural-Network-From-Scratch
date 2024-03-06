#ifndef NODE_H
#define NODE_H

#include <memory>
#include <iostream>
#include "eigen3/Eigen/Eigen"

namespace nn {

enum class Operation {
    add,
    multiply,
};

class Node {
public:
    double value_ = 0;
    double grad_ = 0;
    Operation op_;
    std::vector<std::shared_ptr<Node>> children_;

    Node(double value, Operation op,
         std::vector<std::shared_ptr<Node>> children)
        : value_(value),
          op_(op),
          children_(children) {}
    Node(double value) : value_(value) {}
    Node() {
        SetRandom();
    }
    Node(Node &&) = default;
    Node(const Node &) = default;
    Node &operator=(Node &&) = default;
    Node &operator=(const Node &) = default;

    Node& operator=(const double value) {
        value_ = value;
        return *this;
    }

    Node operator+(const Node& other) const {
        std::vector<std::shared_ptr<Node>> children;
        children.push_back(std::make_shared<Node>(other));
        children.push_back(std::make_shared<Node>(*this));
        return { other.value_ + value_, Operation::add, children };
    }

    void operator+=(const Node& other) {
        *this =  *this + other ;
    }

    /*Node operator-(const Node& other) const {
        std::vector<std::shared_ptr<Node>> children;
        children.push_back(std::make_shared<Node>(other));
        children.push_back(std::make_shared<Node>(*this));
        return { other.value_ - value_, Operation::subtract, children };
    }

    void operator-=(const Node& other) {
        // todo. this won't work
        op_ = Operation::subtract;
        value_ -= other.value_;
        children_.push_back(std::make_shared<Node>(other));
    }*/

    Node operator*(const Node& other) const {
        std::vector<std::shared_ptr<Node>> children;
        children.push_back(std::make_shared<Node>(other));
        children.push_back(std::make_shared<Node>(*this));
        return { other.value_ * value_, Operation::multiply, children };
    }

    Node operator*=(const Node& other) {
        return *this * other;
    }

    /*Node operator/(const Node& other) const {
        std::vector<std::shared_ptr<Node>> children;
        children.push_back(std::make_shared<Node>(other));
        children.push_back(std::make_shared<Node>(*this));
        return { other.value_ / value_, Operation::divide, children };
    }

    void operator/=(const Node& other) {
        value_ /= other.value_;
    }*/

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
                children_[0]->grad_ += children_[1]->grad_;
                children_[1]->grad_ += children_[0]->grad_;
        }
    }

    void ZeroGrad() {
        grad_ = 0;
        for (auto& child : children_) child->ZeroGrad();
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
