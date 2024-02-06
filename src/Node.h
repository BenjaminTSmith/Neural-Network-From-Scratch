#ifndef NODE_H
#define NODE_H

#include <cmath>
#include <vector>

class Node {
public:
    Node(double value) : value(value) {}
    Node() : value(0) {}
    
    Node operator+(const Node& other) const { return {value + other.value}; }
    Node operator+(const int other) const { return {value + other}; }

    Node operator*(const Node& other) const { return {value * other.value}; }
    Node operator*(const int other) const { return {value * other}; }

    void operator+=(const Node& other) {
        value += other.value;
    }

    Node tanh() const { return {(std::exp(2 * value) - 1) / (std::exp(2 * value) + 1)}; }

    double value;
    double grad = 0;

private:

};

#endif // !NODE_H
