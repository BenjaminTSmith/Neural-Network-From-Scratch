#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "eigen3/Eigen/Eigen"

typedef Eigen::VectorXd ColVector;
typedef Eigen::RowVectorXd RowVector;
typedef Eigen::MatrixXd Matrix;

static Matrix Identity(const Matrix& input) {
    return input;
}

static Matrix d_Identity(const Matrix& input) {
    return Matrix::Ones(input.rows(), input.cols());
}

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

static Matrix MSE(const Matrix& input, const Matrix& ground_truth) {
    return (ground_truth.array() - input.array()).cwiseAbs2();
}

#endif // !ACTIVATION_H
