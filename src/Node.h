#ifndef NODE_H
#define NODE_H

#include <cmath>
#include <vector>

class Node {
public:
    Node(double value) : value_(value) {}
    Node() : value_(1) {}
    
    Node operator+(const Node& other) const { return {value_ + other.value_}; }
    Node operator+(const double other) const { return {value_ + other}; }

    Node operator*(const Node& other) const { return {value_ * other.value_}; }
    Node operator*(const double other) const { return {value_ * other}; }

    void operator+=(const Node& other) { value_ += other.value_; }
    void operator+=(const double other) { value_ += other; }

    Node tanh() const { return {(std::exp(2 * value_) - 1) / (std::exp(2 * value_) + 1)}; }

    double value_;
    double grad_ = 0;

private:

};

#endif // !NODE_H
