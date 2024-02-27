#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "eigen3/Eigen/Eigen"

typedef Eigen::VectorXd ColVector;
typedef Eigen::RowVectorXd RowVector;
typedef Eigen::MatrixXd Matrix;

static Matrix ReLU(const Matrix& input) {
    return input.cwiseMax(0);
}

static double LeakyReLU(const double num) {
    return num > 0 ? num : 0.1 * num;
}

static ColVector SoftMax(const ColVector& input) {
    return ColVector(4);
}

#endif // !ACTIVATION_H
