#include <iostream>
#include "Layer.h"

int main() {

    Eigen::MatrixXd input(1, 3);
    input << 7, 8, 9;
    std::cout << input << std::endl;
    Eigen::MatrixXd ground_truth(1, 3);
    ground_truth << 3, -2, -1;

    nn::Layer hidden_layer(5, 3);
    nn::Layer output_layer(1, 5);

    hidden_layer.activation_function_ = ReLU;
    hidden_layer.activation_derivative_ = d_ReLU;
    output_layer.loss_function_ = MSE;
    output_layer.loss_derivative_ = d_MSE;

    hidden_layer.ForwardPass(input);
    output_layer.ForwardPass(hidden_layer);

    return 0;
}
