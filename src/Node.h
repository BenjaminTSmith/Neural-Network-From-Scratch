#ifndef NODE_H
#define NODE_H

#include "eigen3/Eigen/Eigen"

namespace nn {

struct Node {

    double value = 0;
    double delta = 0;

    Node(double value) : value(value) {}

    Node operator+(const Node& other) const { return {value + other.value}; }
    Node operator+(const double other) const { return {value + other}; }

    Node operator-(const Node& other) const { return {value - other.value}; }
    Node operator-(const double other) const { return {value - other}; }

    Node operator*(const Node& other) const { return {value * other.value}; }
    Node operator*(const double other) const { return {value * other}; }

    Node operator/(const Node& other) const { return {value / other.value}; }
    Node operator/(const double other) const { return {value / other}; }

    bool operator>(const double other) const { return value > other; }
    bool operator<(const double other) const { return value < other; }

    void operator+=(const Node& other) { value += other.value; }
    void operator+=(const double other) { value += other; }

    void operator()() { value -= delta; }

};

}

namespace Eigen {
template<> struct NumTraits<nn::Node> {
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
        MulCost = 1,
    };
}; 
}

#endif // !NODE_H
