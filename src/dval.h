#ifndef DVAL_H
#define DVAL_H

#include <cstdlib>
#include <iostream>
#include <random>
#include <chrono>

struct Dval {
    double value_;
    double grad_;

    Dval(double value, double grad = 0) : value_(value), grad_(grad) {}

    Dval(const Dval& other) : value_(other.value_), grad_(other.grad_) {}

    Dval() : grad_(0) { SetRandom(); }

    Dval& operator=(const Dval& other) = default;
    Dval& operator=(double value) {
        value_ = value;
        grad_ = 0;
        return *this;
    };

    Dval operator+(const Dval& other) const {
        return {value_ + other.value_, grad_ + other.grad_};
    }
    Dval operator*(const Dval& other) const { return {value_ * other.value_}; }
    Dval operator-(const Dval& other) const { return {value_ - other.value_}; }
    Dval operator/(const Dval& other) const { return {value_ / other.value_}; }
    Dval& operator+=(const Dval& other) {
        value_ += other.value_;
        return *this;
    }
    Dval& operator*=(const Dval& other) {
        value_ *= other.value_;
        return *this;
    }
    Dval& operator-=(const Dval& other) {
        value_ -= other.value_;
        return *this;
    }
    Dval& operator/=(const Dval& other) {
        value_ /= other.value_;
        return *this;
    }

    Dval operator+(double other) { return {value_ + other}; }
    Dval operator*(double other) { return {value_ * other}; }
    Dval operator-(double other) { return {value_ - other}; }
    Dval operator/(double other) { return {value_ / other}; }
    Dval& operator+=(double other) {
        value_ += other;
        return *this;
    }
    Dval& operator*=(double other) {
        value_ *= other;
        return *this;
    }
    Dval& operator-=(double other) {
        value_ -= other;
        return *this;
    }
    Dval& operator/=(double other) {
        value_ /= other;
        return *this;
    }

    bool operator<(const Dval& other) const { return value_ < other.value_; }
    bool operator>(const Dval& other) const { return value_ > other.value_; }

    void SetRandom() {
        std::default_random_engine generator;
        generator.seed(time(0));
        std::uniform_real_distribution<double> distribution(-1, 1);
        value_ = distribution(generator);
    }

    explicit operator double() const { return value_; }
};

static std::ostream& operator<<(std::ostream& oss, const Dval& dval) {
    oss << dval.value_;
    return oss;
}

#endif // !DVAL_H
