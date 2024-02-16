#ifndef NODE_H
#define NODE_H

#include <cmath>
#include <math.h>
#include <vector>
#include <iostream>
#include <random>

class Node {
public:
    Node(double value) : value_(value) {}
    Node() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(-1.0, 1.0);

        value_ = dis(gen);
    }
    
    Node operator+(const Node& other) const { return {value_ + other.value_}; }
    Node operator+(const double other) const { return {value_ + other}; }

    Node operator*(const Node& other) const { return {value_ * other.value_}; }
    Node operator*(const double other) const { return {value_ * other}; }

    void operator+=(const Node& other) { value_ += other.value_; }
    void operator+=(const double other) { value_ += other; }

    Node tanh() const { 
        double value = (std::exp(2 * value_) - 1) / (std::exp(2 * value_) + 1);
        return {(std::exp(2 * value_) - 1) / (std::exp(2 * value_) + 1)}; 
    }

    Node ReLU() const {
        return {value_ > 0 ? value_ : 0};
    }

    double value_;
    double grad_ = 0;

private:

};

#endif // !NODE_H
