#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <algorithm>
#include "eigen3/Eigen/Eigen"

typedef Eigen::VectorXd ColVector;

static double ReLU(const double num) {
    return std::max(0.0, num);
}

static double LeakyReLU(const double num) {
    return num > 0 ? num : 0.1 * num;
}

static ColVector SoftMax(const ColVector& input) {
    return ColVector(4);
}

#endif // !ACTIVATION_H
