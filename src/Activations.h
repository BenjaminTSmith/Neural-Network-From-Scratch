#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "eigen3/Eigen/Eigen"

typedef Eigen::VectorXd ColVector;
typedef Eigen::RowVectorXd RowVector;
typedef Eigen::MatrixXd Matrix;

static Matrix ReLU(const Matrix& input) {
    return input.cwiseMax(0);
}

static Matrix d_ReLU(const Matrix& input) {
    return input.unaryExpr([](double num) { return num > 0 ? 1 : 0; })
        .cast<double>();
}

static Matrix LeakyReLU(const Matrix& input) {
    return input.unaryExpr([](double num) {
        return num > 0 ? num : 0.1 * num; 
    }).cast<double>();
}

static Matrix d_LeakyReLU(const Matrix& input) {
    return input.unaryExpr([](double num) { return num > 0 ? 1 : 0.1; })
        .cast<double>();
}

static ColVector SoftMax(const ColVector& input) {
    return ColVector(4);
}

#endif // !ACTIVATION_H
