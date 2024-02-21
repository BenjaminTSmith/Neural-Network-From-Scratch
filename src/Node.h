#ifndef NODE_H
#define NODE_H

#include <random>

struct Node {

    double value_ = 0;
    double grad_ = 0;

    Node(double value) : value_(value) {}

    Node() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(-1.0, 1.0);

        value_ = dis(gen);
    }
    
    Node operator+(const Node& other) const { return {value_ + other.value_}; }
    Node operator+(const double other) const { return {value_ + other}; }

    Node operator-(const Node& other) const { return {value_ - other.value_}; }
    Node operator-(const double other) const { return {value_ - other}; }

    Node operator*(const Node& other) const { return {value_ * other.value_}; }
    Node operator*(const double other) const { return {value_ * other}; }

    bool operator>(const double other) const { return value_ > other; }
    bool operator<(const double other) const { return value_ < other; }

    void operator+=(const Node& other) { value_ += other.value_; }
    void operator+=(const double other) { value_ += other; }

};

#endif // !NODE_H
