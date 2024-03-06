#ifndef NODE_H
#define NODE_H

#include "eigen3/Eigen/Eigen"

namespace nn {

class Node {
public:
    Eigen::MatrixXd value_;
    Eigen::MatrixXd delta_;

    Node(const Eigen::MatrixXd value) : value_(value) {}
    Node(const Eigen::MatrixXd value, const Eigen::MatrixXd delta)
        : value_(value), delta_(delta) {}
    Node(int rows, int cols) : value_(rows, cols), delta_(rows, cols) {}
    Node() {}

    Node(const Node& copy) 
        : value_(copy.value_),
          delta_(copy.delta_) {}

    void ZeroGrad() {
        delta_ = Eigen::MatrixXd::Zero(delta_.rows(), delta_.cols()); 
        delta_.fill(0);
    }

    double mean() { return value_.mean(); }

    Node operator+(const Node& other) const { 
        return { value_ + other.value_ }; 
    }

    Node operator+(const double other) const {
        return { value_.array() + other }; 
    }

    Node operator-(const Node& other) const {
        return { value_ - other.value_ }; 
    }

    Node operator-(const double other) const {
        return { value_.array() - other }; 
    }

    Node operator*(const Node& other) const {
        return { value_ * other.value_ }; 
    }

    Node operator*(const double other) const { return { value_ * other }; }

    Node& operator+=(const Node& other) {
        value_ += other.value_; 
        return *this;
    }

    Node& operator+=(const double other) {
        value_.array() += other; 
        return *this;
    }

    Node& operator-=(const Node& other) {
        value_ -= other.value_; 
        return *this;
    }

    Node& operator-=(const double other) {
        value_.array() -= other; 
        return *this;
    }

    Node& operator*=(const Node& other) {
        value_ *= other.value_; 
        return *this;
    }

    Node& operator*=(const double other) {
        value_ *= other; 
        return *this;
    }

    Node& operator/=(const Node& other) {
        value_.array() /= other.value_.array(); 
        return *this;
    }

    Node& operator/=(const double other) {
        value_ /= other; 
        return *this;
    }

    void operator()() { value_ -= delta_; }

};

}

#endif // !NODE_H
