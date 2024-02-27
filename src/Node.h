#ifndef NODE_H
#define NODE_H

#include "eigen3/Eigen/Core"

namespace nn {

struct Node {

    double value = 0;
    double delta = 0;

    Node(double value) : value(value) {}
    Node() {}

    Node operator+(const Node& other) const { return { value + other.value }; }

    template<typename T>
    Node operator+(const T other) const { return { value + other }; }

    Node operator-(const Node& other) const { return { value - other.value }; }

    template<typename T>
    Node operator-(const T other) const { return { value - other }; }

    Node operator*(const Node& other) const { return { value * other.value }; }

    template<typename T>
    Node operator*(const T other) const { return { value * other }; }

    Node operator/(const Node& other) const { return { value / other.value }; }

    template<typename T>
    Node operator/(const T other) const { return { value / other }; }

    Node& operator+=(const Node& other) {
        value += other.value; 
        return *this;
    }

    template<typename T>
    Node& operator+=(const T other) {
        value += other; 
        return *this;
    }

    Node& operator-=(const Node& other) {
        value -= other.value; 
        return *this;
    }

    template<typename T>
    Node& operator-=(const T other) {
        value -= other; 
        return *this;
    }

    Node& operator*=(const Node& other) {
        value *= other.value; 
        return *this;
    }

    template<typename T>
    Node& operator*=(const T other) {
        value *= other; 
        return *this;
    }

    Node& operator/=(const Node& other) {
        value /= other.value; 
        return *this;
    }

    template<typename T>
    Node& operator/=(const T other) {
        value /= other; 
        return *this;
    }


    bool operator>(const double other) const { return value > other; }
    bool operator<(const double other) const { return value < other; }

    void operator()() { value -= delta; }

};

}

namespace Eigen {
template<> struct NumTraits<nn::Node>: NumTraits<double> {
    typedef nn::Node Real;
    typedef nn::Node NonInteger;
    typedef nn::Node Nested;

    enum {
        IsComplex = 0,
        IsInteger = 0,
        IsSigned = 1,
        RequireInitialization = 1,
        ReadCost = 1,
        AddCost = 3,
        MulCost = 3,
    };
}; 

template<typename BinaryOp>
struct ScalarBinaryOpTraits<nn::Node, double, BinaryOp> {
    typedef nn::Node ReturnType;
};

template<typename BinaryOp>
struct ScalarBinaryOpTraits<double, nn::Node, BinaryOp> {
    typedef nn::Node ReturnType;
};

}

#endif // !NODE_H
