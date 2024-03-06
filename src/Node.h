#ifndef NODE_H
#define NODE_H

#include <memory>
#include "eigen3/Eigen/Eigen"

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

    Node operator*(const Node& other) const {
        std::vector<std::shared_ptr<Node>> children;
        children.push_back(std::make_shared<Node>(other));
        children.push_back(std::make_shared<Node>(*this));
        return { other.value_ * value_, Operation::multiply, children };
    }

    void operator*=(const Node& other) {
        *this = *this * other;
    }

    void ReLU() {
        std::vector<std::shared_ptr<Node>> child;
        child.push_back(std::make_shared<Node>(*this));
        *this = { value_ > 0 ? value_ : 0, Operation::ReLU, child };
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
                children_[0]->grad_ += children_[1]->grad_ * grad_;
                children_[1]->grad_ += children_[0]->grad_ * grad_;
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
